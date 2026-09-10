#include "rl_sdk/metric_catalog.h"
#include "maze/task_entry.h"
#include "task/inference/onnx_inferencer.h"
#include "log/logger.h"

#include <grpcpp/grpcpp.h>
#include <cstdio>
#include <string>
#include <csignal>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <limits>
#include <sstream>
#include <thread>
#include <filesystem>
#include <vector>
#include <unistd.h>
#include "proto/metrics/catalog.grpc.pb.h"

// --- 全局信号标志 ---
static std::atomic<bool> g_running{true};

// ---- 信号处理（优雅退出）----
static void SignalHandler(int sig) {
    g_running.store(false);
}

// --- 默认配置文件路径 ---
static const char* kDefaultConfigPath = "configs/server_config.yaml";
static const char* kManagedReadyMarker =
    "/run/rl/aiserver-managed-ready";

static bool PublishManagedReadyMarker(
    const common::ServiceInstanceIdentity& source,
    int container_port,
    std::string& error) {
    const char* managed = std::getenv("RL_INFRA_MANAGED");
    if (managed == nullptr || std::string(managed) != "true") return true;
    namespace fs = std::filesystem;
    std::error_code filesystem_error;
    const fs::path destination(kManagedReadyMarker);
    fs::create_directories(destination.parent_path(), filesystem_error);
    if (filesystem_error) {
        error = "cannot create managed readiness directory: " +
                filesystem_error.message();
        return false;
    }
    const fs::path temporary =
        destination.string() + ".tmp." + std::to_string(getpid());
    {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output) {
            error = "cannot open managed readiness marker";
            return false;
        }
        output << "component=" << source.component() << "\n"
               << "instance_id=" << source.instance_id() << "\n"
               << "lifecycle_epoch=" << source.lifecycle_epoch() << "\n"
               << "container_port=" << container_port << "\n";
        output.flush();
        if (!output) {
            fs::remove(temporary, filesystem_error);
            error = "cannot flush managed readiness marker";
            return false;
        }
    }
    fs::rename(temporary, destination, filesystem_error);
    if (filesystem_error) {
        fs::remove(temporary, filesystem_error);
        error = "cannot publish managed readiness marker: " +
                filesystem_error.message();
        return false;
    }
    return true;
}

static void RemoveManagedReadyMarker() {
    const char* managed = std::getenv("RL_INFRA_MANAGED");
    if (managed == nullptr || std::string(managed) != "true") return;
    std::error_code ignored;
    std::filesystem::remove(kManagedReadyMarker, ignored);
}

static void PrintUsage() {
    std::fputs(
        "Usage: rl_aiserver [options]\n"
        "\n"
        "Configuration is resolved once as CLI > allowlisted environment > "
        "config.\n"
        "--config and --inspect-model are diagnostic/meta options; every "
        "business\n"
        "override below replaces the named field in the selected config.\n"
        "\n"
        "  --config PATH                    select the YAML config file\n"
        "  --workload training|evaluation   -> server.run_mode\n"
        "  --listen-port PORT               -> server.listen_port\n"
        "  --evaluation-model PATH          -> "
        "model.evaluation_model_path\n"
        "  --sample-distributor HOST:PORT   -> "
        "sample_distributor.host/port\n"
        "  --model-distributor HOST:PORT    -> "
        "model_distribution.host/port\n"
        "  --inspect-model PATH --observation-dim N --action-count N\n"
        "                                   validate one SaveModel.onnx and exit\n"
        "  --help, -h                       show this help and exit\n",
        stdout);
}

static std::string ShapeJson(const std::vector<int64_t>& shape) {
    std::ostringstream output;
    output << "[";
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index > 0) output << ",";
        output << shape[index];
    }
    output << "]";
    return output.str();
}

static bool ParsePositiveDimension(const std::string& value, int& result) {
    try {
        std::size_t consumed = 0;
        const long long candidate = std::stoll(value, &consumed);
        if (consumed != value.size() || candidate <= 0 ||
            candidate > std::numeric_limits<int>::max()) {
            return false;
        }
        result = static_cast<int>(candidate);
        return true;
    } catch (...) {
        return false;
    }
}

static int InspectModel(const std::filesystem::path& model_path,
                        int observation_dimension,
                        int action_count) {
    std::error_code filesystem_error;
    if (std::filesystem::is_symlink(model_path, filesystem_error) ||
        filesystem_error ||
        !std::filesystem::is_regular_file(model_path, filesystem_error) ||
        filesystem_error) {
        std::fprintf(
            stderr,
            "--inspect-model must name an explicit regular ONNX file\n");
        return 2;
    }

    Logger::Instance().SetConsoleLevel(LogLevel::ERROR);
    OnnxInferencer inferencer;
    OnnxInferencer::PreparedModel prepared;
    std::string error;
    if (!inferencer.PrepareModel(
            model_path.string(), observation_dimension, action_count,
            prepared, &error)) {
        std::fprintf(stderr, "model inspection failed: %s\n", error.c_str());
        return 2;
    }

    const auto& session = prepared.session;
    const auto input_shape = session->GetInputTypeInfo(0)
                                 .GetTensorTypeAndShapeInfo()
                                 .GetShape();
    const auto action_shape = session->GetOutputTypeInfo(0)
                                  .GetTensorTypeAndShapeInfo()
                                  .GetShape();
    const auto value_shape = session->GetOutputTypeInfo(1)
                                 .GetTensorTypeAndShapeInfo()
                                 .GetShape();
    std::cout
        << "{\"observation_dimension\":" << observation_dimension
        << ",\"action_count\":" << action_count
        << ",\"input\":{\"name\":\"observation\",\"dtype\":\"float32\","
        << "\"shape\":" << ShapeJson(input_shape) << "},"
        << "\"action_output\":{\"name\":\"action_logits\","
        << "\"dtype\":\"float32\",\"shape\":"
        << ShapeJson(action_shape) << "},"
        << "\"value_output\":{\"name\":\"value\","
        << "\"dtype\":\"float32\",\"shape\":"
        << ShapeJson(value_shape) << "}}"
        << std::endl;
    return 0;
}

static bool ParsePort(const std::string& value, int& port) {
    try {
        std::size_t consumed = 0;
        const int candidate = std::stoi(value, &consumed);
        if (consumed != value.size() || candidate <= 0 || candidate > 65535) {
            return false;
        }
        port = candidate;
        return true;
    } catch (...) {
        return false;
    }
}

static bool ParseAddress(const std::string& value,
                         std::string& host,
                         int& port) {
    const std::size_t separator = value.rfind(':');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 >= value.size()) {
        return false;
    }
    int parsed_port = 0;
    if (!ParsePort(value.substr(separator + 1), parsed_port)) {
        return false;
    }
    host = value.substr(0, separator);
    port = parsed_port;
    return true;
}

static bool ParseNonNegativeInt64(const std::string& value,
                                  int64_t& result) {
    try {
        std::size_t consumed = 0;
        const int64_t candidate = std::stoll(value, &consumed);
        if (consumed != value.size() || candidate < 0) return false;
        result = candidate;
        return true;
    } catch (...) {
        return false;
    }
}

struct ParsedCommandLine {
    std::string config_path = kDefaultConfigPath;
    aiserver::TaskEntry::Overrides overrides;
};

static bool ParseCommandLine(int argc,
                             char* argv[],
                             ParsedCommandLine& parsed,
                             std::string& error) {
    bool config_seen = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                error = std::string(name) + " requires a value";
                return nullptr;
            }
            return argv[++i];
        };

        if (argument == "--config") {
            const char* candidate = value("--config");
            if (!candidate) return false;
            if (config_seen) {
                error = "--config may be specified only once";
                return false;
            }
            config_seen = true;
            parsed.config_path = candidate;
        } else if (argument == "--workload") {
            const char* candidate = value("--workload");
            if (!candidate) return false;
            if (parsed.overrides.workload.has_value()) {
                error = "--workload may be specified only once";
                return false;
            }
            const int workload = aiserver_mode::Parse(candidate);
            if (!aiserver_mode::IsValid(workload)) {
                error = "unknown workload: " + std::string(candidate);
                return false;
            }
            parsed.overrides.workload = workload;
        } else if (argument == "--listen-port") {
            const char* candidate = value("--listen-port");
            int port = 0;
            if (!candidate || parsed.overrides.listen_port.has_value() ||
                !ParsePort(candidate, port)) {
                error = "--listen-port must be a valid TCP port";
                return false;
            }
            parsed.overrides.listen_port = port;
        } else if (argument == "--model-distributor") {
            const char* candidate = value("--model-distributor");
            std::string host;
            int port = 0;
            if (!candidate ||
                parsed.overrides.model_distributor_host.has_value() ||
                !ParseAddress(candidate, host, port)) {
                error = "--model-distributor must use host:port";
                return false;
            }
            parsed.overrides.model_distributor_host = host;
            parsed.overrides.model_distributor_port = port;
        } else if (argument == "--sample-distributor") {
            const char* candidate = value("--sample-distributor");
            std::string host;
            int port = 0;
            if (!candidate ||
                parsed.overrides.sample_distributor_host.has_value() ||
                !ParseAddress(candidate, host, port)) {
                error = "--sample-distributor must use host:port";
                return false;
            }
            parsed.overrides.sample_distributor_host = host;
            parsed.overrides.sample_distributor_port = port;
        } else if (argument == "--evaluation-model") {
            const char* candidate = value("--evaluation-model");
            if (!candidate ||
                parsed.overrides.evaluation_model_path.has_value()) {
                error = "--evaluation-model may be specified only once";
                return false;
            }
            parsed.overrides.evaluation_model_path = candidate;
        } else if (argument.rfind("--", 0) == 0) {
            error = "unknown argument: " + argument;
            return false;
        } else {
            error = "positional arguments are not supported: " + argument;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc == 2 &&
        (std::string(argv[1]) == "--help" ||
         std::string(argv[1]) == "-h")) {
        PrintUsage();
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--inspect-model") {
        if (argc < 3 || (argc - 3) % 2 != 0) {
            std::fputs(
                "--inspect-model requires PATH, --observation-dim N, and "
                "--action-count N\n",
                stderr);
            return 2;
        }
        int observation_dimension = 0;
        int action_count = 0;
        for (int index = 3; index < argc; index += 2) {
            const std::string option(argv[index]);
            const std::string value(argv[index + 1]);
            if (option == "--observation-dim" &&
                observation_dimension == 0 &&
                ParsePositiveDimension(value, observation_dimension)) {
                continue;
            }
            if (option == "--action-count" && action_count == 0 &&
                ParsePositiveDimension(value, action_count)) {
                continue;
            }
            std::fprintf(stderr, "invalid model inspection option: %s\n",
                         option.c_str());
            return 2;
        }
        if (observation_dimension <= 0 || action_count <= 0) {
            std::fputs(
                "--inspect-model requires --observation-dim N and "
                "--action-count N\n",
                stderr);
            return 2;
        }
        return InspectModel(
            argv[2], observation_dimension, action_count);
    }

    std::printf("============================================\n");
    std::printf("  RL Training Framework - AIServer\n");
    std::printf("============================================\n\n");

    // ---- 0. 初始化日志系统 ----
    Logger::Instance().Init("log");
    Logger::Instance().SetConsoleLevel(LogLevel::INFO);
    Logger::Instance().SetFileLevel(LogLevel::DEBUG);

    // ---- 1. 注册信号处理 ----
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    ParsedCommandLine parsed;
    std::string argument_error;
    if (!ParseCommandLine(argc, argv, parsed, argument_error)) {
        LOG_ERROR("Main", "命令参数无效: %s", argument_error.c_str());
        Logger::Instance().Close();
        return 2;
    }

    aiserver::TaskEntry::Config cfg;
    aiserver::TaskEntry::LoadReport load_report;
    std::string config_error;
    if (!aiserver::TaskEntry::LoadConfig(parsed.config_path, parsed.overrides, cfg,
                          load_report, config_error)) {
        LOG_ERROR("Main", "配置加载或校验失败: %s (%s)",
                  parsed.config_path.c_str(),
                  config_error.empty() ? "see config diagnostics"
                                       : config_error.c_str());
        Logger::Instance().Close();
        return 2;
    }
    LOG_INFO("Main", "config source: %s", load_report.config_path.c_str());
    const auto log_overrides = [](const char* source,
                                  const std::vector<std::string>& values) {
        if (values.empty()) {
            LOG_INFO("Main", "%s overrides: none", source);
            return;
        }
        std::ostringstream fields;
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (index > 0) fields << ",";
            fields << values[index];
        }
        LOG_INFO("Main", "%s overrides: %s", source,
                 fields.str().c_str());
    };
    log_overrides("environment", load_report.environment_overridden_fields);
    log_overrides("CLI", load_report.cli_overridden_fields);
    LOG_INFO(
        "Main",
        "最终配置: workload=%s, listen=0.0.0.0:%d, "
        "evaluation_model=%s, local_train=%s, "
        "model_distributor=%s:%d, sample_distributor=%s:%d, "
        "max_agents=%d",
        aiserver_mode::Workload(cfg.server.run_mode),
        cfg.server.listen_port,
        cfg.model.evaluation_model_path.c_str(),
        cfg.model.local_train_dir.c_str(),
        cfg.model_distribution.host.c_str(),
        cfg.model_distribution.port,
        cfg.sample_distributor.host.c_str(),
        cfg.sample_distributor.port, cfg.server.max_agents);
    aiserver::TaskEntry::LogConfig(cfg);

    // ---- 3. 创建 gRPC 服务 ----
    aiserver::TaskEntry::Service service(cfg);
    if (!service.Start()) {
        LOG_ERROR("Main", "AIServer 启动失败，详见上方错误");
        Logger::Instance().Close();
        return 1;
    }

    std::string listen_addr = "0.0.0.0:" + std::to_string(cfg.server.listen_port);

    rl_sdk::MetricCatalogService metric_catalog([&] { return service.MetricCatalog(); });
    grpc::ServerBuilder builder;
    builder.AddListeningPort(listen_addr, grpc::InsecureServerCredentials());
    builder.RegisterService(
        static_cast<aiserver::TaskEntry::Service::RpcService*>(&service));
    if (aiserver_mode::ExposesTrainingStatus(cfg.server.run_mode)) {
        builder.RegisterService(&metric_catalog);
        builder.RegisterService(
            static_cast<training::AIServerTrainingStatusService::Service*>(
                &service));
        builder.RegisterService(
            &service.Metrics());
    }

    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();

    if (!server) {
        LOG_ERROR("Main", "gRPC 服务启动失败，端口: %s", listen_addr.c_str());
        return 1;
    }

    std::string readiness_error;
    if (!PublishManagedReadyMarker(
            service.MetricSourceIdentity(), cfg.server.listen_port,
            readiness_error)) {
        LOG_ERROR("Main", "AIServer managed readiness 发布失败: %s",
                  readiness_error.c_str());
        service.BeginShutdown();
        server->Shutdown();
        server->Wait();
        Logger::Instance().Close();
        return 1;
    }

    LOG_INFO("Main", "AIServer 已启动，监听: %s", listen_addr.c_str());
    LOG_INFO("Main", "运行模式: %d (%s)", cfg.server.run_mode,
             aiserver_mode::Workload(cfg.server.run_mode));
    LOG_INFO("Main", "等待 Client 连接...");

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    LOG_INFO("Main", "收到停止信号，开始清理样本链路");
    const bool shutdown_clean = service.BeginShutdown();
    RemoveManagedReadyMarker();
    server->Shutdown(
        std::chrono::system_clock::now() + std::chrono::seconds(2));
    server->Wait();

    if (shutdown_clean) {
        LOG_INFO("Main", "AIServer 已停止");
    } else {
        LOG_ERROR(
            "Main",
            "AIServer 停止失败: 样本处置未收敛，详见 TrainingTaskService 错误日志");
    }
    Logger::Instance().Close();
    return shutdown_clean ? 0 : 1;
}
