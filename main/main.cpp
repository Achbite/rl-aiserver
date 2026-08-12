#include "grpc/maze_service.h"
#include "config/config_loader.h"
#include "log/logger.h"

#include <grpcpp/grpcpp.h>
#include <cstdio>
#include <string>
#include <csignal>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

// --- 全局信号标志 ---
static std::atomic<bool> g_running{true};

// ---- 信号处理（优雅退出）----
static void SignalHandler(int sig) {
    g_running.store(false);
}

// --- 默认配置文件路径 ---
static const char* kDefaultConfigPath = "configs/server_config.yaml";

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

static bool ApplyCommandLine(int argc,
                             char* argv[],
                             AIServerConfig& config,
                             std::string& config_path,
                             std::string& error) {
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
            config_path = candidate;
        } else if (argument == "--workload") {
            const char* candidate = value("--workload");
            if (!candidate) return false;
            config.server.run_mode = aiserver_mode::Parse(candidate);
            if (!aiserver_mode::IsValid(config.server.run_mode)) {
                error = "unknown workload: " + std::string(candidate);
                return false;
            }
        } else if (argument == "--listen-port") {
            const char* candidate = value("--listen-port");
            if (!candidate || !ParsePort(candidate, config.server.listen_port)) {
                error = "--listen-port must be a valid TCP port";
                return false;
            }
        } else if (argument == "--model-distributor") {
            const char* candidate = value("--model-distributor");
            if (!candidate ||
                !ParseAddress(candidate,
                              config.model_distribution.host,
                              config.model_distribution.port)) {
                error = "--model-distributor must use host:port";
                return false;
            }
        } else if (argument == "--sample-distributor") {
            const char* candidate = value("--sample-distributor");
            if (!candidate ||
                !ParseAddress(candidate,
                              config.sample_output.host,
                              config.sample_output.port)) {
                error = "--sample-distributor must use host:port";
                return false;
            }
        } else if (argument == "--local-train-dir") {
            const char* candidate = value("--local-train-dir");
            if (!candidate) return false;
            config.model.local_train_dir = candidate;
        } else if (argument == "--local-test-model-dir" ||
                   argument == "--smoke-model-dir") {
            const char* candidate = value(argument.c_str());
            if (!candidate) return false;
            config.model.local_test_dir = candidate;
        } else if (argument == "--train") {
            config.server.run_mode = aiserver_mode::kTraining;
        } else if (argument.rfind("--", 0) == 0) {
            error = "unknown argument: " + argument;
            return false;
        } else {
            config_path = argument;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    std::printf("============================================\n");
    std::printf("  迷宫训练框架 - AIServer\n");
    std::printf("============================================\n\n");

    // ---- 0. 初始化日志系统 ----
    Logger::Instance().Init("log");
    Logger::Instance().SetConsoleLevel(LogLevel::INFO);
    Logger::Instance().SetFileLevel(LogLevel::DEBUG);

    // ---- 1. 注册信号处理 ----
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    std::string config_path = kDefaultConfigPath;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (argument == "--workload" ||
                   argument == "--listen-port" ||
                   argument == "--model-distributor" ||
                   argument == "--sample-distributor" ||
                   argument == "--local-train-dir" ||
                   argument == "--local-test-model-dir" ||
                   argument == "--smoke-model-dir") {
            ++i;
        } else if (argument.rfind("--", 0) != 0) {
            config_path = argument;
        }
    }
    AIServerConfig cfg;
    if (!LoadServerConfig(config_path, cfg)) {
        LOG_ERROR("Main", "配置加载或 0.10.0 身份校验失败: %s",
                  config_path.c_str());
        Logger::Instance().Close();
        return 2;
    }

    std::string argument_error;
    if (!ApplyCommandLine(
            argc, argv, cfg, config_path, argument_error)) {
        LOG_ERROR("Main", "命令参数无效: %s", argument_error.c_str());
        Logger::Instance().Close();
        return 2;
    }
    LOG_INFO(
        "Main",
        "最终配置: workload=%d, listen=0.0.0.0:%d, "
        "model_distributor=%s:%d, sample_distributor=%s:%d",
        cfg.server.run_mode,
        cfg.server.listen_port,
        cfg.model_distribution.host.c_str(),
        cfg.model_distribution.port,
        cfg.sample_output.host.c_str(),
        cfg.sample_output.port);

    // ---- 3. 创建 gRPC 服务 ----
    MazeServiceImpl service(cfg);
    if (!service.Start()) {
        LOG_ERROR("Main", "AIServer 启动失败，详见上方错误");
        Logger::Instance().Close();
        return 1;
    }

    std::string listen_addr = "0.0.0.0:" + std::to_string(cfg.server.listen_port);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(listen_addr, grpc::InsecureServerCredentials());
    builder.RegisterService(
        static_cast<maze::MazeTaskService::Service*>(&service));
    if (aiserver_mode::ExposesTrainingStatus(cfg.server.run_mode)) {
        builder.RegisterService(
            static_cast<training::AIServerTrainingStatusService::Service*>(
                &service));
    }

    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();

    if (!server) {
        LOG_ERROR("Main", "gRPC 服务启动失败，端口: %s", listen_addr.c_str());
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
    server->Shutdown(
        std::chrono::system_clock::now() + std::chrono::seconds(2));
    server->Wait();

    if (shutdown_clean) {
        LOG_INFO("Main", "AIServer 已停止");
    } else {
        LOG_ERROR(
            "Main",
            "AIServer 停止失败: 样本处置未收敛，详见 MazeService 错误日志");
    }
    Logger::Instance().Close();
    return shutdown_clean ? 0 : 1;
}
