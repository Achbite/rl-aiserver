#include "maze/config/config.h"

#include "log/logger.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <vector>
#include <cctype>
#include <cmath>
#include <limits>

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
    try {
        std::size_t consumed = 0;
        const int value = std::stoi(val, &consumed);
        return consumed == val.size() ? value
                                      : std::numeric_limits<int>::min();
    } catch (...) {
        return std::numeric_limits<int>::min();
    }
}

static int64_t SafeInt64(const std::string& val, int64_t def) {
    if (val.empty()) return def;
    try {
        std::size_t consumed = 0;
        const int64_t value = std::stoll(val, &consumed);
        return consumed == val.size()
                   ? value
                   : std::numeric_limits<int64_t>::min();
    } catch (...) {
        return std::numeric_limits<int64_t>::min();
    }
}

static std::size_t SafeSize(const std::string& val, std::size_t def) {
    if (val.empty()) return def;
    try {
        std::size_t consumed = 0;
        const auto value = std::stoull(val, &consumed);
        return consumed == val.size()
                   ? static_cast<std::size_t>(value)
                   : std::numeric_limits<std::size_t>::max();
    } catch (...) {
        return std::numeric_limits<std::size_t>::max();
    }
}

static double SafeDouble(const std::string& val, double def) {
    if (val.empty()) return def;
    try {
        std::size_t consumed = 0;
        const double value = std::stod(val, &consumed);
        return consumed == val.size() ? value : def;
    } catch (...) {
        return def;
    }
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

static bool IsStrictBool(const std::string& value) {
    if (value.empty()) return false;
    std::string normalized = value;
    for (char& character : normalized) {
        character = static_cast<char>(
            std::tolower(static_cast<unsigned char>(character)));
    }
    return normalized == "true" || normalized == "false" ||
           normalized == "1" || normalized == "0" ||
           normalized == "yes" || normalized == "no" ||
           normalized == "on" || normalized == "off";
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

static bool ReadEnvironment(const char* name,
                            std::optional<std::string>& value,
                            std::string& error) {
    const char* raw = std::getenv(name);
    if (!raw) return true;
    if (*raw == '\0') {
        error = std::string(name) + " must not be empty";
        return false;
    }
    const std::string candidate(raw);
    if (candidate != Trim(candidate)) {
        error = std::string(name) +
                " must not contain surrounding whitespace";
        return false;
    }
    value = candidate;
    return true;
}

static bool ReadEnvironmentInt(const char* name,
                               std::optional<int>& value,
                               std::string& error) {
    std::optional<std::string> raw;
    if (!ReadEnvironment(name, raw, error) || !raw.has_value()) {
        return error.empty();
    }
    const int parsed = SafeInt(*raw, std::numeric_limits<int>::min());
    if (parsed == std::numeric_limits<int>::min()) {
        error = std::string(name) + " must be an integer";
        return false;
    }
    value = parsed;
    return true;
}

// ---- 从 YAML 文件加载配置，依次应用环境和 CLI 覆盖 ----
bool LoadMazeConfig(const std::string& yaml_path,
                      const MazeConfigOverrides& overrides,
                      MazeConfig& out_config,
                      MazeConfigLoadReport& report,
                      std::string& error) {
    namespace fs = std::filesystem;
    out_config = MazeConfig{};
    report = MazeConfigLoadReport{};
    error.clear();

    std::error_code fs_error;
    fs::path config_path = fs::absolute(fs::path(yaml_path), fs_error);
    if (fs_error) {
        error = "cannot resolve config path: " + yaml_path;
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }
    config_path = fs::weakly_canonical(config_path, fs_error);
    if (fs_error) {
        error = "cannot canonicalize config path: " + yaml_path;
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }
    report.config_path = config_path.string();

    std::ifstream ifs(config_path);
    if (!ifs.is_open()) {
        error = "cannot open config file: " + config_path.string();
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string content = ss.str();
    ifs.close();

    LOG_INFO("Config", "加载配置文件: %s", config_path.c_str());

    std::vector<YamlEntry> entries = ParseYaml(content);
    static const std::set<std::string> allowed_entries = {
        "server.run_mode", "server.listen_port", "server.max_agents",
        "policy.training_temperature", "policy.sampling_seed",
        "policy.action_mask_mode",
        "rollout.gamma", "rollout.gae_lambda", "rollout.tmax", "metrics.reward_interval_ms",
        "observation.ray_max_range", "model.evaluation_model_path",
        "model.local_train_dir", "model.startup_timeout_ms",
        "model.expected_obs_dim", "model.expected_action_dim",
        "environment.agent_count", "task.fixed_map_id",
        "task.episode_max_steps",
        "model_distribution.host", "model_distribution.port",
        "model_distribution.poll_interval_ms",
        "model_distribution.rpc_timeout_ms", "sample_distributor.enabled",
        "sample_distributor.host", "sample_distributor.port",
        "sample_distributor.envelope_max_transitions",
        "sample_distributor.envelope_max_bytes",
        "sample_distributor.rpc_timeout_ms",
        "sample_distributor.max_attempts",
        "sample_distributor.enqueue_timeout_ms",
        "sample_distributor.drain_timeout_ms",
        "sample_distributor.health_timeout_ms",
        "sample_distributor.status_poll_interval_ms",
        "sample_distributor.recovery_timeout_ms",
        "sample_distributor.outbound_max_envelopes",
        "sample_distributor.outbound_max_estimated_bytes",
        "sample_distributor.aiserver_id", "sample_distributor.env_id",
    };
    for (const auto& entry : entries) {
        const std::string qualified = entry.section + "." + entry.key;
        if (allowed_entries.count(qualified) == 0) {
            error = "unknown AIServer config field: " + qualified;
            LOG_ERROR("Config", "%s", error.c_str());
            return false;
        }
    }

    const std::pair<const char*, const char*> required[] = {
        {"policy", "training_temperature"},
        {"policy", "sampling_seed"},
        {"policy", "action_mask_mode"},
        {"rollout", "gamma"},
        {"rollout", "gae_lambda"},
        {"rollout", "tmax"},
        {"observation", "ray_max_range"},
        {"server", "run_mode"},
        {"server", "listen_port"},
        {"server", "max_agents"},
        {"environment", "agent_count"},
        {"task", "fixed_map_id"},
        {"task", "episode_max_steps"},
        {"model", "evaluation_model_path"},
        {"model", "local_train_dir"},
        {"model", "expected_obs_dim"},
        {"model", "expected_action_dim"},
        {"model_distribution", "host"},
        {"model_distribution", "port"},
        {"sample_distributor", "host"},
        {"sample_distributor", "port"},
        {"sample_distributor", "recovery_timeout_ms"},
    };
    for (const auto& field : required) {
        if (FindValue(entries, field.first, field.second).empty()) {
            LOG_ERROR("Config", "缺少关键配置: %s.%s",
                      field.first, field.second);
            return false;
        }
    }

    const std::pair<const char*, const char*> integer_fields[] = {
        {"metrics", "reward_interval_ms"},
        {"policy", "sampling_seed"},
        {"rollout", "tmax"},
        {"observation", "ray_max_range"},
        {"server", "listen_port"},
        {"server", "max_agents"},
        {"environment", "agent_count"},
        {"model", "startup_timeout_ms"},
        {"model", "expected_obs_dim"},
        {"model", "expected_action_dim"},
        {"task", "episode_max_steps"},
        {"model_distribution", "port"},
        {"model_distribution", "poll_interval_ms"},
        {"model_distribution", "rpc_timeout_ms"},
        {"sample_distributor", "port"},
        {"sample_distributor", "envelope_max_transitions"},
        {"sample_distributor", "envelope_max_bytes"},
        {"sample_distributor", "rpc_timeout_ms"},
        {"sample_distributor", "max_attempts"},
        {"sample_distributor", "enqueue_timeout_ms"},
        {"sample_distributor", "drain_timeout_ms"},
        {"sample_distributor", "health_timeout_ms"},
        {"sample_distributor", "status_poll_interval_ms"},
        {"sample_distributor", "recovery_timeout_ms"},
        {"sample_distributor", "outbound_max_envelopes"},
        {"sample_distributor", "outbound_max_estimated_bytes"},
    };
    for (const auto& field : integer_fields) {
        const std::string value = FindValue(entries, field.first, field.second);
        if (!value.empty() &&
            (SafeInt64(value, 0) == std::numeric_limits<int64_t>::min() ||
             value.front() == '-')) {
            LOG_ERROR("Config", "整数配置无效: %s.%s",
                      field.first, field.second);
            return false;
        }
    }
    const std::string sample_distributor_enabled =
        FindValue(entries, "sample_distributor", "enabled");
    if (!sample_distributor_enabled.empty() &&
        !IsStrictBool(sample_distributor_enabled)) {
        LOG_ERROR("Config", "布尔配置无效: sample_distributor.enabled");
        return false;
    }

    out_config.reward_metric_interval_ms = SafeInt(FindValue(entries, "metrics", "reward_interval_ms"), 5000);
    if (out_config.reward_metric_interval_ms <= 0) {
        error = "metrics.reward_interval_ms must be positive";
        return false;
    }
    out_config.policy.training_temperature = SafeDouble(
        FindValue(entries, "policy", "training_temperature"), -1.0);
    out_config.policy.sampling_seed = static_cast<uint32_t>(SafeSize(
        FindValue(entries, "policy", "sampling_seed"), 0));
    out_config.policy.action_mask_mode =
        FindValue(entries, "policy", "action_mask_mode");
    out_config.rollout.gamma =
        SafeDouble(FindValue(entries, "rollout", "gamma"), -1.0);
    out_config.rollout.gae_lambda =
        SafeDouble(FindValue(entries, "rollout", "gae_lambda"), -1.0);
    out_config.rollout.tmax = static_cast<uint32_t>(SafeSize(
        FindValue(entries, "rollout", "tmax"), 0));
    out_config.observation.ray_max_range = SafeInt(
        FindValue(entries, "observation", "ray_max_range"), 0);

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
            LOG_ERROR("Config", "运行模式无效: %s",
                      configured_mode.c_str());
            return false;
        }
    }

    // --- model ---
    out_config.model.evaluation_model_path =
        FindValue(entries, "model", "evaluation_model_path");
    std::string local_train_dir =
        FindValue(entries, "model", "local_train_dir");
    if (!local_train_dir.empty()) {
        out_config.model.local_train_dir = local_train_dir;
    }
    out_config.model.startup_timeout_ms =
        SafeInt(FindValue(entries, "model", "startup_timeout_ms"), 30000);
    out_config.model.expected_obs_dim = SafeInt(
        FindValue(entries, "model", "expected_obs_dim"), 0);
    out_config.model.expected_action_dim = SafeInt(
        FindValue(entries, "model", "expected_action_dim"), 0);

    // --- task ---
    out_config.environment.agent_count = SafeInt(
        FindValue(entries, "environment", "agent_count"), 4);
    std::string fixed_map_id =
        FindValue(entries, "task", "fixed_map_id");
    if (!fixed_map_id.empty()) out_config.task.fixed_map_id = fixed_map_id;
    out_config.task.episode_max_steps = SafeInt(
        FindValue(entries, "task", "episode_max_steps"), 0);

    std::string model_host =
        FindValue(entries, "model_distribution", "host");
    if (!model_host.empty()) {
        out_config.model_distribution.host = model_host;
    }
    out_config.model_distribution.port = SafeInt(
        FindValue(entries, "model_distribution", "port"), 9200);
    out_config.model_distribution.poll_interval_ms = SafeInt(
        FindValue(entries, "model_distribution", "poll_interval_ms"), 200);
    out_config.model_distribution.rpc_timeout_ms = SafeInt(
        FindValue(entries, "model_distribution", "rpc_timeout_ms"), 5000);

    // --- AIServer-local asynchronous SampleDistributor ---
    out_config.sample_distributor.enabled = SafeBool(
        FindValue(entries, "sample_distributor", "enabled"), true);
    std::string shost = FindValue(entries, "sample_distributor", "host");
    if (!shost.empty()) {
        out_config.sample_distributor.host = shost;
    }
    out_config.sample_distributor.port = SafeInt(
        FindValue(entries, "sample_distributor", "port"), 9100);
    out_config.sample_distributor.envelope_max_transitions = SafeInt(
        FindValue(entries, "sample_distributor", "envelope_max_transitions"),
        128);
    out_config.sample_distributor.envelope_max_bytes = SafeSize(
        FindValue(entries, "sample_distributor", "envelope_max_bytes"),
        8ULL * 1024ULL * 1024ULL);
    out_config.sample_distributor.rpc_timeout_ms = SafeInt(
        FindValue(entries, "sample_distributor", "rpc_timeout_ms"), 2000);
    out_config.sample_distributor.max_attempts = SafeInt(
        FindValue(entries, "sample_distributor", "max_attempts"), 4);
    out_config.sample_distributor.enqueue_timeout_ms = SafeInt(
        FindValue(entries, "sample_distributor", "enqueue_timeout_ms"), 100);
    out_config.sample_distributor.drain_timeout_ms = SafeInt(
        FindValue(entries, "sample_distributor", "drain_timeout_ms"), 10000);
    out_config.sample_distributor.health_timeout_ms = SafeInt(
        FindValue(entries, "sample_distributor", "health_timeout_ms"), 5000);
    out_config.sample_distributor.status_poll_interval_ms = SafeInt(
        FindValue(entries, "sample_distributor", "status_poll_interval_ms"),
                200);
    out_config.sample_distributor.recovery_timeout_ms = SafeInt(
        FindValue(entries, "sample_distributor", "recovery_timeout_ms"),
        30000);
    out_config.sample_distributor.outbound_max_envelopes = SafeSize(
        FindValue(entries, "sample_distributor", "outbound_max_envelopes"),
        64);
    out_config.sample_distributor.outbound_max_estimated_bytes = SafeSize(
        FindValue(entries, "sample_distributor", "outbound_max_estimated_bytes"),
        64ULL * 1024ULL * 1024ULL);
    std::string aiserver_id =
        FindValue(entries, "sample_distributor", "aiserver_id");
    if (!aiserver_id.empty()) {
        out_config.sample_distributor.aiserver_id = aiserver_id;
    }
    std::string env_id =
        FindValue(entries, "sample_distributor", "env_id");
    if (!env_id.empty()) {
        out_config.sample_distributor.env_id = env_id;
    }
    const auto record_environment_override = [&](const char* field) {
        report.environment_overridden_fields.emplace_back(field);
    };
    std::optional<int> environment_integer;
    std::optional<std::string> environment_string;
    if (!ReadEnvironmentInt("RL_AISERVER_MAX_AGENTS", environment_integer,
                            error)) {
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }
    if (environment_integer.has_value()) {
        out_config.server.max_agents = *environment_integer;
        record_environment_override("server.max_agents");
    }
    environment_integer.reset();
    if (!ReadEnvironmentInt("RL_AISERVER_AGENT_COUNT", environment_integer,
                            error)) {
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }
    if (environment_integer.has_value()) {
        out_config.environment.agent_count = *environment_integer;
        record_environment_override("environment.agent_count");
    }
    environment_string.reset();
    if (!ReadEnvironment("RL_INFRA_DATA_ROOT", environment_string, error)) {
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }
    if (environment_string.has_value()) {
        const fs::path data_root(*environment_string);
        if (!data_root.is_absolute()) {
            error = "RL_INFRA_DATA_ROOT must be absolute";
            LOG_ERROR("Config", "%s", error.c_str());
            return false;
        }
        out_config.model.local_train_dir =
            (data_root / "aiserver").lexically_normal().string();
        record_environment_override("model.local_train_dir");
    }
    environment_string.reset();
    if (!ReadEnvironment("RL_INFRA_POD_ID", environment_string, error)) {
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }
    if (environment_string.has_value()) {
        out_config.sample_distributor.aiserver_id =
            *environment_string + "-aiserver";
        out_config.sample_distributor.env_id =
            *environment_string + "-env";
        record_environment_override("sample_distributor.aiserver_id");
        record_environment_override("sample_distributor.env_id");
    }
    environment_string.reset();
    if (!ReadEnvironment("RL_TASK_MAP_ID", environment_string, error)) {
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }
    if (environment_string.has_value()) {
        out_config.task.fixed_map_id = *environment_string;
        record_environment_override("task.fixed_map_id");
    }
    environment_integer.reset();
    if (!ReadEnvironmentInt("RL_TASK_EPISODE_MAX_STEPS",
                            environment_integer, error)) {
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }
    if (environment_integer.has_value()) {
        out_config.task.episode_max_steps = *environment_integer;
        record_environment_override("task.episode_max_steps");
    }

    const auto record_cli_override = [&](const char* field) {
        report.cli_overridden_fields.emplace_back(field);
    };
    if (overrides.sample_distributor_host.has_value() !=
            overrides.sample_distributor_port.has_value() ||
        overrides.model_distributor_host.has_value() !=
            overrides.model_distributor_port.has_value()) {
        error = "distributor CLI overrides require a complete host:port";
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }
    if (overrides.workload.has_value()) {
        out_config.server.run_mode = *overrides.workload;
        record_cli_override("server.run_mode");
    }
    if (overrides.listen_port.has_value()) {
        out_config.server.listen_port = *overrides.listen_port;
        record_cli_override("server.listen_port");
    }
    if (overrides.evaluation_model_path.has_value()) {
        out_config.model.evaluation_model_path =
            *overrides.evaluation_model_path;
        record_cli_override("model.evaluation_model_path");
    }
    if (overrides.sample_distributor_host.has_value()) {
        out_config.sample_distributor.host = *overrides.sample_distributor_host;
        out_config.sample_distributor.port = *overrides.sample_distributor_port;
        record_cli_override("sample_distributor.address");
    }
    if (overrides.model_distributor_host.has_value()) {
        out_config.model_distribution.host =
            *overrides.model_distributor_host;
        out_config.model_distribution.port =
            *overrides.model_distributor_port;
        record_cli_override("model_distribution.address");
    }

    const auto resolve_config_path = [&](std::string& value) {
        fs::path path(value);
        if (path.is_relative()) path = config_path.parent_path() / path;
        value = fs::absolute(path).lexically_normal().string();
    };
    resolve_config_path(out_config.model.evaluation_model_path);
    resolve_config_path(out_config.model.local_train_dir);

    if (overrides.evaluation_model_path.has_value() &&
        out_config.server.run_mode == aiserver_mode::kTraining) {
        error = "--evaluation-model is invalid for the training workload";
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }
    if (out_config.server.run_mode == aiserver_mode::kEvaluation) {
        const fs::path model_path(out_config.model.evaluation_model_path);
        const auto model_status = fs::symlink_status(model_path, fs_error);
        if (out_config.model.evaluation_model_path.empty() || fs_error ||
            fs::is_symlink(model_status) ||
            !fs::is_regular_file(model_status)) {
            error = "evaluation requires an explicit regular, non-symlink "
                    "model file";
            LOG_ERROR("Config", "%s: %s", error.c_str(),
                      model_path.c_str());
            return false;
        }
    }

    static const std::regex map_id_pattern("[A-Za-z0-9_-]+");
    if (!std::regex_match(out_config.task.fixed_map_id, map_id_pattern) ||
        out_config.task.episode_max_steps <= 0) {
        LOG_ERROR("Config", "effective Maze map or episode identity is invalid");
        return false;
    }
    if (out_config.model.expected_obs_dim <= 0 ||
        out_config.model.expected_action_dim <= 0 ||
        !std::isfinite(out_config.policy.training_temperature) ||
        out_config.policy.training_temperature <= 0.0 ||
        (out_config.policy.action_mask_mode != "disabled" &&
         out_config.policy.action_mask_mode != "required") ||
        out_config.observation.ray_max_range <= 0) {
        LOG_ERROR("Config", "model, policy or observation config is invalid");
        return false;
    }
    const bool runtime_values_valid =
        out_config.server.listen_port > 0 &&
        out_config.server.listen_port <= 65535 &&
        out_config.server.max_agents > 0 &&
        out_config.environment.agent_count > 0 &&
        out_config.server.max_agents >= out_config.environment.agent_count &&
        aiserver_mode::IsValid(out_config.server.run_mode) &&
        std::isfinite(out_config.rollout.gamma) &&
        out_config.rollout.gamma >= 0.0 &&
        out_config.rollout.gamma <= 1.0 &&
        std::isfinite(out_config.rollout.gae_lambda) &&
        out_config.rollout.gae_lambda >= 0.0 &&
        out_config.rollout.gae_lambda <= 1.0 &&
        out_config.rollout.tmax > 0 &&
        out_config.model.startup_timeout_ms > 0 &&
        out_config.model_distribution.port > 0 &&
        out_config.model_distribution.port <= 65535 &&
        out_config.model_distribution.poll_interval_ms > 0 &&
        out_config.model_distribution.rpc_timeout_ms > 0 &&
        out_config.sample_distributor.port > 0 &&
        out_config.sample_distributor.port <= 65535 &&
        out_config.sample_distributor.envelope_max_transitions > 0 &&
        out_config.sample_distributor.envelope_max_bytes > 0 &&
        out_config.sample_distributor.rpc_timeout_ms > 0 &&
        out_config.sample_distributor.max_attempts > 0 &&
        out_config.sample_distributor.enqueue_timeout_ms > 0 &&
        out_config.sample_distributor.drain_timeout_ms > 0 &&
        out_config.sample_distributor.health_timeout_ms > 0 &&
        out_config.sample_distributor.status_poll_interval_ms > 0 &&
        out_config.sample_distributor.recovery_timeout_ms > 0 &&
        out_config.sample_distributor.outbound_max_envelopes > 0 &&
        out_config.sample_distributor.outbound_max_estimated_bytes > 0;
    if (!runtime_values_valid) {
        LOG_ERROR("Config", "运行时数值配置无效");
        return false;
    }
    LOG_INFO("Config", "server: port=%d, max_agents=%d, run_mode=%d(%s)",
             out_config.server.listen_port, out_config.server.max_agents,
             out_config.server.run_mode,
             aiserver_mode::Workload(out_config.server.run_mode));
    LOG_INFO("Config", "model: evaluation_model=%s, local_train=%s, startup_timeout_ms=%d, shape=[%d]->[%d], mask=%s",
             out_config.model.evaluation_model_path.c_str(),
             out_config.model.local_train_dir.c_str(),
             out_config.model.startup_timeout_ms,
             out_config.model.expected_obs_dim,
             out_config.model.expected_action_dim,
             out_config.policy.action_mask_mode.c_str());
    LOG_INFO("Config", "environment: agent_count=%d; task: map=%s",
             out_config.environment.agent_count,
             out_config.task.fixed_map_id.c_str());
    LOG_INFO("Config", "model_distribution: target=%s:%d, poll_interval_ms=%d, rpc_timeout_ms=%d",
             out_config.model_distribution.host.c_str(),
             out_config.model_distribution.port,
             out_config.model_distribution.poll_interval_ms,
             out_config.model_distribution.rpc_timeout_ms);
    const bool sample_distributor_active =
        out_config.server.run_mode == aiserver_mode::kTraining &&
        out_config.sample_distributor.enabled;
    LOG_INFO("Config", "sample_distributor: configured_enabled=%s, active=%s, ingress=%s:%d, envelope_max=%d/%zuB, rpc_timeout_ms=%d, status_poll_interval_ms=%d, recovery_timeout_ms=%d, attempts=%d, local_queue=%zu/%zuB, aiserver_id=%s, env_id=%s",
             out_config.sample_distributor.enabled ? "true" : "false",
             sample_distributor_active ? "true" : "false",
             out_config.sample_distributor.host.c_str(),
             out_config.sample_distributor.port,
             out_config.sample_distributor.envelope_max_transitions,
             out_config.sample_distributor.envelope_max_bytes,
             out_config.sample_distributor.rpc_timeout_ms,
             out_config.sample_distributor.status_poll_interval_ms,
             out_config.sample_distributor.recovery_timeout_ms,
             out_config.sample_distributor.max_attempts,
             out_config.sample_distributor.outbound_max_envelopes,
             out_config.sample_distributor.outbound_max_estimated_bytes,
             out_config.sample_distributor.aiserver_id.c_str(),
             out_config.sample_distributor.env_id.c_str());
    error.clear();
    return true;
}

bool LoadMazeConfig(const std::string& yaml_path,
                      MazeConfig& out_config) {
    MazeConfigLoadReport report;
    std::string error;
    return LoadMazeConfig(
        yaml_path, MazeConfigOverrides{}, out_config, report, error);
}
