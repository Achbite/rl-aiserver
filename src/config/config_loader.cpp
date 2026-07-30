#include "config/config_loader.h"
#include "log/logger.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <cctype>
#include <cstdlib>

// ---- 去除字符串首尾空白 ----
static std::string Trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end   = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

// ---- 去除字符串值的引号包裹 ----
static std::string StripQuotes(const std::string& s) {
    if (s.size() >= 2 &&
        ((s.front() == '"' && s.back() == '"') ||
         (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

// ---- 安全转换辅助 ----
static int SafeInt(const std::string& val, int def) {
    if (val.empty()) return def;
    try { return std::stoi(val); } catch (...) { return def; }
}

static std::size_t SafeSize(const std::string& val, std::size_t def) {
    if (val.empty()) return def;
    try { return static_cast<std::size_t>(std::stoull(val)); } catch (...) { return def; }
}

static bool SafeBool(const std::string& val, bool def) {
    if (val.empty()) return def;
    std::string v = val;
    for (char& ch : v) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    if (v == "true" || v == "1" || v == "yes" || v == "on") return true;
    if (v == "false" || v == "0" || v == "no" || v == "off") return false;
    return def;
}

static std::string GetEnvValue(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : "";
}

static int EnvInt(const char* name, int def) {
    return SafeInt(GetEnvValue(name), def);
}

// ---- YAML 键值对 ----
struct YamlEntry {
    std::string section;
    std::string key;
    std::string value;
};

// ---- 解析 YAML 文本为键值对列表 ----
static std::vector<YamlEntry> ParseYaml(const std::string& text) {
    std::vector<YamlEntry> entries;
    std::istringstream stream(text);
    std::string line;
    std::string current_section;

    while (std::getline(stream, line)) {
        // 去除行内注释
        size_t comment_pos = std::string::npos;
        bool in_quotes = false;
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '"' || line[i] == '\'') {
                in_quotes = !in_quotes;
            } else if (line[i] == '#' && !in_quotes) {
                comment_pos = i;
                break;
            }
        }
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        std::string trimmed = Trim(line);
        if (trimmed.empty()) continue;

        size_t colon_pos = trimmed.find(':');
        if (colon_pos == std::string::npos) continue;

        std::string key_part = Trim(trimmed.substr(0, colon_pos));
        std::string val_part = Trim(trimmed.substr(colon_pos + 1));

        bool has_indent = (!line.empty() && (line[0] == ' ' || line[0] == '\t'));

        if (!has_indent && val_part.empty()) {
            current_section = key_part;
        } else if (has_indent && !key_part.empty()) {
            YamlEntry entry;
            entry.section = current_section;
            entry.key     = key_part;
            entry.value   = StripQuotes(val_part);
            entries.push_back(entry);
        }
    }

    return entries;
}

// ---- 在解析结果中查找指定 section.key 的值 ----
static std::string FindValue(const std::vector<YamlEntry>& entries,
                             const std::string& section,
                             const std::string& key) {
    for (const auto& e : entries) {
        if (e.section == section && e.key == key) {
            return e.value;
        }
    }
    return "";
}

// ---- 从 YAML 文件加载配置 ----
bool LoadServerConfig(const std::string& yaml_path, AIServerConfig& out_config) {
    std::ifstream ifs(yaml_path);
    if (!ifs.is_open()) {
        LOG_WARN("Config", "无法打开配置文件: %s，使用默认值", yaml_path.c_str());
        out_config = AIServerConfig{};
        return false;
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string content = ss.str();
    ifs.close();

    LOG_INFO("Config", "加载配置文件: %s", yaml_path.c_str());

    std::vector<YamlEntry> entries = ParseYaml(content);

    // --- server ---
    out_config.server.listen_port = SafeInt(FindValue(entries, "server", "listen_port"), 9002);
    out_config.server.max_agents  = SafeInt(FindValue(entries, "server", "max_agents"),  10);
    const std::string configured_mode =
        FindValue(entries, "server", "run_mode");
    if (!configured_mode.empty()) {
        const int parsed_mode = aiserver_mode::Parse(configured_mode);
        if (aiserver_mode::IsValid(parsed_mode)) {
            out_config.server.run_mode = parsed_mode;
        } else {
            LOG_WARN("Config", "忽略未知运行模式: %s",
                     configured_mode.c_str());
        }
    }

    // --- strategy ---
    out_config.strategy.grid_size        = SafeInt(FindValue(entries, "strategy", "grid_size"),        500);
    out_config.strategy.replan_interval  = SafeInt(FindValue(entries, "strategy", "replan_interval"),  10);

    // --- model ---
    std::string local_dir = FindValue(entries, "model", "local_dir");
    if (!local_dir.empty()) {
        out_config.model.local_dir = local_dir;
    }
    std::string local_train_dir =
        FindValue(entries, "model", "local_train_dir");
    if (!local_train_dir.empty()) {
        out_config.model.local_train_dir = local_train_dir;
    }
    std::string smoke_dir = FindValue(entries, "model", "smoke_dir");
    if (!smoke_dir.empty()) {
        out_config.model.smoke_dir = smoke_dir;
    }
    std::string save_name = FindValue(entries, "model", "save_name");
    if (!save_name.empty()) {
        out_config.model.save_name = save_name;
    }
    std::string manifest_name = FindValue(entries, "model", "manifest_name");
    if (!manifest_name.empty()) {
        out_config.model.manifest_name = manifest_name;
    }
    out_config.model.startup_timeout_ms =
        SafeInt(FindValue(entries, "model", "startup_timeout_ms"), 30000);
    out_config.model.expected_obs_dim =
        SafeInt(FindValue(entries, "model", "expected_obs_dim"), 13);
    out_config.model.expected_action_dim =
        SafeInt(FindValue(entries, "model", "expected_action_dim"), 9);

    std::string model_host =
        FindValue(entries, "model_distribution", "host");
    if (!model_host.empty()) {
        out_config.model_distribution.host = model_host;
    }
    out_config.model_distribution.port = SafeInt(
        FindValue(entries, "model_distribution", "port"), 9200);
    out_config.model_distribution.poll_interval_ms = SafeInt(
        FindValue(entries, "model_distribution", "poll_interval_ms"), 200);
    out_config.model_distribution.boundary_wait_ms = SafeInt(
        FindValue(entries, "model_distribution", "boundary_wait_ms"), 1000);
    out_config.model_distribution.rpc_timeout_ms = SafeInt(
        FindValue(entries, "model_distribution", "rpc_timeout_ms"), 5000);
    std::string contract_version =
        FindValue(entries, "model_distribution", "contract_version");
    if (!contract_version.empty()) {
        out_config.model_distribution.contract_version = contract_version;
    }

    // --- sample_output ---
    out_config.sample_output.enabled = SafeBool(FindValue(entries, "sample_output", "enabled"), true);
    std::string shost = FindValue(entries, "sample_output", "host");
    if (!shost.empty()) {
        out_config.sample_output.host = shost;
    }
    out_config.sample_output.port = SafeInt(FindValue(entries, "sample_output", "port"), 9100);
    out_config.sample_output.fragment_samples =
        SafeInt(FindValue(entries, "sample_output", "fragment_samples"), 128);
    out_config.sample_output.rpc_timeout_ms =
        SafeInt(FindValue(entries, "sample_output", "rpc_timeout_ms"), 2000);
    out_config.sample_output.max_attempts =
        SafeInt(FindValue(entries, "sample_output", "max_attempts"), 4);
    out_config.sample_output.enqueue_timeout_ms =
        SafeInt(FindValue(entries, "sample_output", "enqueue_timeout_ms"), 100);
    out_config.sample_output.drain_timeout_ms =
        SafeInt(FindValue(entries, "sample_output", "drain_timeout_ms"), 10000);
    out_config.sample_output.health_timeout_ms =
        SafeInt(FindValue(entries, "sample_output", "health_timeout_ms"), 5000);
    out_config.sample_output.outbound_max_fragments =
        SafeSize(FindValue(entries, "sample_output", "outbound_max_fragments"), 64);
    out_config.sample_output.outbound_max_estimated_bytes =
        SafeSize(FindValue(entries, "sample_output", "outbound_max_estimated_bytes"),
                 64ULL * 1024ULL * 1024ULL);
    std::string aiserver_id = FindValue(entries, "sample_output", "aiserver_id");
    if (!aiserver_id.empty()) {
        out_config.sample_output.aiserver_id = aiserver_id;
    }
    std::string env_id = FindValue(entries, "sample_output", "env_id");
    if (!env_id.empty()) {
        out_config.sample_output.env_id = env_id;
    }
    out_config.metrics.episode_window = SafeSize(
        FindValue(entries, "metrics", "episode_window"), 100);

    std::string listen_port = GetEnvValue("MAZE_LISTEN_PORT");
    if (!listen_port.empty()) {
        out_config.server.listen_port = SafeInt(listen_port, out_config.server.listen_port);
    }
    std::string aiserver_port = GetEnvValue("MAZE_AISERVER_PORT");
    if (!aiserver_port.empty()) {
        out_config.server.listen_port = SafeInt(aiserver_port, out_config.server.listen_port);
    }
    std::string run_mode = GetEnvValue("MAZE_AISERVER_RUN_MODE");
    if (!run_mode.empty()) {
        const int candidate = aiserver_mode::Parse(run_mode);
        if (aiserver_mode::IsValid(candidate)) {
            out_config.server.run_mode = candidate;
        } else {
            LOG_WARN("Config", "忽略未知环境运行模式: %s",
                     run_mode.c_str());
        }
    }
    std::string model_distributor_host =
        GetEnvValue("MAZE_MODEL_DISTRIBUTOR_HOST");
    if (!model_distributor_host.empty()) {
        out_config.model_distribution.host = model_distributor_host;
    }
    std::string model_distributor_port =
        GetEnvValue("MAZE_MODEL_DISTRIBUTOR_PORT");
    if (!model_distributor_port.empty()) {
        out_config.model_distribution.port = SafeInt(
            model_distributor_port, out_config.model_distribution.port);
    }
    std::string local_train_root =
        GetEnvValue("MAZE_LOCAL_TRAIN_ROOT");
    if (!local_train_root.empty()) {
        out_config.model.local_train_dir = local_train_root;
    }
    std::string smoke_model_dir =
        GetEnvValue("MAZE_SMOKE_MODEL_DIR");
    if (!smoke_model_dir.empty()) {
        out_config.model.smoke_dir = smoke_model_dir;
    }
    std::string local_model_dir =
        GetEnvValue("MAZE_LOCAL_MODEL_DIR");
    if (!local_model_dir.empty()) {
        out_config.model.local_dir = local_model_dir;
    }
    std::string sd_host = GetEnvValue("MAZE_SAMPLE_DISTRIBUTOR_HOST");
    if (!sd_host.empty()) {
        out_config.sample_output.host = sd_host;
    }
    std::string sd_port = GetEnvValue("MAZE_SAMPLE_DISTRIBUTOR_PORT");
    if (!sd_port.empty()) {
        out_config.sample_output.port = SafeInt(sd_port, out_config.sample_output.port);
    }
    std::string env_aiserver_id = GetEnvValue("MAZE_AISERVER_ID");
    if (!env_aiserver_id.empty()) {
        out_config.sample_output.aiserver_id = env_aiserver_id;
    }
    std::string env_env_id = GetEnvValue("MAZE_ENV_ID");
    if (!env_env_id.empty()) {
        out_config.sample_output.env_id = env_env_id;
    }
    out_config.sample_output.fragment_samples =
        EnvInt("MAZE_SAMPLE_FRAGMENT_SIZE", out_config.sample_output.fragment_samples);
    out_config.sample_output.rpc_timeout_ms =
        EnvInt("MAZE_SAMPLE_RPC_TIMEOUT_MS", out_config.sample_output.rpc_timeout_ms);
    out_config.sample_output.max_attempts =
        EnvInt("MAZE_SAMPLE_MAX_ATTEMPTS", out_config.sample_output.max_attempts);
    out_config.sample_output.enqueue_timeout_ms =
        EnvInt("MAZE_SAMPLE_ENQUEUE_TIMEOUT_MS", out_config.sample_output.enqueue_timeout_ms);
    out_config.sample_output.drain_timeout_ms =
        EnvInt("MAZE_SAMPLE_DRAIN_TIMEOUT_MS", out_config.sample_output.drain_timeout_ms);
    out_config.model.startup_timeout_ms =
        EnvInt("MAZE_MODEL_STARTUP_TIMEOUT_MS", out_config.model.startup_timeout_ms);
    out_config.model_distribution.poll_interval_ms =
        EnvInt(
            "MAZE_MODEL_POLL_INTERVAL_MS",
            out_config.model_distribution.poll_interval_ms);
    out_config.model_distribution.boundary_wait_ms =
        EnvInt(
            "MAZE_MODEL_BOUNDARY_WAIT_MS",
            out_config.model_distribution.boundary_wait_ms);
    out_config.metrics.episode_window = SafeSize(
        GetEnvValue("MAZE_EPISODE_METRICS_WINDOW"),
        out_config.metrics.episode_window);
    if (out_config.metrics.episode_window == 0) {
        out_config.metrics.episode_window = 100;
    }

    LOG_INFO("Config", "server: port=%d, max_agents=%d, run_mode=%d(%s)",
             out_config.server.listen_port, out_config.server.max_agents,
             out_config.server.run_mode,
             aiserver_mode::Workload(out_config.server.run_mode));
    LOG_INFO("Config", "strategy: grid=%d, replan=%d",
             out_config.strategy.grid_size,
             out_config.strategy.replan_interval);
    LOG_INFO("Config", "model: local=%s, local_train=%s, smoke=%s, manifest=%s, startup_timeout_ms=%d, shape=[%d]->[%d]",
             out_config.model.local_dir.c_str(),
             out_config.model.local_train_dir.c_str(),
             out_config.model.smoke_dir.c_str(),
             out_config.model.manifest_name.c_str(),
             out_config.model.startup_timeout_ms,
             out_config.model.expected_obs_dim,
             out_config.model.expected_action_dim);
    LOG_INFO("Config", "model_distribution: target=%s:%d, poll_interval_ms=%d, boundary_wait_ms=%d, rpc_timeout_ms=%d, contract=%s",
             out_config.model_distribution.host.c_str(),
             out_config.model_distribution.port,
             out_config.model_distribution.poll_interval_ms,
             out_config.model_distribution.boundary_wait_ms,
             out_config.model_distribution.rpc_timeout_ms,
             out_config.model_distribution.contract_version.c_str());
    LOG_INFO("Config", "sample_output: enabled=%s, target=%s:%d, fragment=%d, rpc_timeout_ms=%d, attempts=%d, queue=%zu/%zuB, aiserver_id=%s, env_id=%s",
             out_config.sample_output.enabled ? "true" : "false",
             out_config.sample_output.host.c_str(),
             out_config.sample_output.port,
             out_config.sample_output.fragment_samples,
             out_config.sample_output.rpc_timeout_ms,
             out_config.sample_output.max_attempts,
             out_config.sample_output.outbound_max_fragments,
             out_config.sample_output.outbound_max_estimated_bytes,
             out_config.sample_output.aiserver_id.c_str(),
             out_config.sample_output.env_id.c_str());
    LOG_INFO("Config", "metrics: episode_window=%zu",
             out_config.metrics.episode_window);
    return true;
}
