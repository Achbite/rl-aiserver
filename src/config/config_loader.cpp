#include "config/config_loader.h"
#include "log/logger.h"

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>
#include <openssl/evp.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>

namespace {

bool ReadFile(const std::filesystem::path& path,
              std::string& bytes,
              std::string& error) {
    std::ifstream stream(path, std::ios::binary);
    std::ostringstream content;
    content << stream.rdbuf();
    if (!stream.good() && !stream.eof()) {
        error = "cannot read file: " + path.string();
        return false;
    }
    bytes = content.str();
    return true;
}

std::string TrimAscii(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string Sha256Hex(const std::string& bytes) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    const bool digest_ok = context &&
        EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
        EVP_DigestUpdate(context, bytes.data(), bytes.size()) == 1 &&
        EVP_DigestFinal_ex(context, digest.data(), &digest_size) == 1;
    if (context) EVP_MD_CTX_free(context);
    if (!digest_ok) return "";
    std::ostringstream hex;
    hex << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index) {
        hex << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    return hex.str();
}

const google::protobuf::Value* JsonField(
    const google::protobuf::Struct& document,
    const std::string& name) {
    const auto found = document.fields().find(name);
    return found == document.fields().end() ? nullptr : &found->second;
}

const google::protobuf::Struct* JsonStruct(
    const google::protobuf::Struct& document,
    const std::string& name) {
    const auto* value = JsonField(document, name);
    return value && value->kind_case() == google::protobuf::Value::kStructValue
        ? &value->struct_value()
        : nullptr;
}

bool JsonStringEquals(const google::protobuf::Struct& document,
                      const std::string& name,
                      const std::string& expected) {
    const auto* value = JsonField(document, name);
    return value && value->kind_case() == google::protobuf::Value::kStringValue &&
           value->string_value() == expected;
}

bool JsonNumberEquals(const google::protobuf::Struct& document,
                      const std::string& name,
                      double expected) {
    const auto* value = JsonField(document, name);
    return value && value->kind_case() == google::protobuf::Value::kNumberValue &&
           value->number_value() == expected;
}

std::string JsonQuote(const std::string& value) {
    std::ostringstream output;
    output << '"';
    static constexpr char kHex[] = "0123456789abcdef";
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u00" << kHex[character >> 4]
                           << kHex[character & 0x0f];
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    output << '"';
    return output.str();
}

std::string CanonicalFileTableDigest(
    const google::protobuf::Struct& files) {
    std::map<std::string, std::string> ordered;
    for (const auto& item : files.fields()) {
        if (item.second.kind_case() !=
                google::protobuf::Value::kStringValue ||
            item.second.string_value().empty()) {
            return "";
        }
        ordered.emplace(item.first, item.second.string_value());
    }
    if (ordered.empty()) return "";
    std::ostringstream canonical;
    canonical << '{';
    bool first = true;
    for (const auto& item : ordered) {
        if (!first) canonical << ',';
        first = false;
        canonical << JsonQuote(item.first) << ':'
                  << JsonQuote(item.second);
    }
    canonical << '}';
    return Sha256Hex(canonical.str());
}

bool LoadMetricEventSchema(const std::string& yaml_path,
                           const std::string& configured_path,
                           const ContractConfig& contract,
                           MetricsConfig& metrics,
                           std::string& error) {
    namespace fs = std::filesystem;
    fs::path catalog_path(configured_path);
    if (catalog_path.is_relative()) {
        catalog_path = fs::path(yaml_path).parent_path() / catalog_path;
    }
    std::error_code fs_error;
    catalog_path = fs::weakly_canonical(catalog_path, fs_error);
    if (fs_error || !fs::is_regular_file(catalog_path, fs_error)) {
        error = "metric schema catalog is missing: " +
                catalog_path.string();
        return false;
    }
    std::string bytes;
    if (!ReadFile(catalog_path, bytes, error)) return false;
    google::protobuf::Struct document;
    const auto status = google::protobuf::util::JsonStringToMessage(
        bytes, &document);
    const auto schema_id = document.fields().find("schema_id");
    const auto schema_version = document.fields().find("schema_version");
    if (!status.ok() || schema_id == document.fields().end() ||
        schema_version == document.fields().end() ||
        schema_id->second.kind_case() !=
            google::protobuf::Value::kStringValue ||
        schema_version->second.kind_case() !=
            google::protobuf::Value::kNumberValue ||
        schema_id->second.string_value() != "maze.metrics.v2" ||
        schema_version->second.number_value() != 2.0) {
        error = "metric schema catalog identity is invalid";
        return false;
    }
    const std::string catalog_digest = Sha256Hex(bytes);
    if (catalog_digest.empty()) {
        error = "cannot calculate metric schema catalog digest";
        return false;
    }

    fs::path digest_path = catalog_path;
    digest_path.replace_extension(".sha256");
    std::string digest_bytes;
    if (!ReadFile(digest_path, digest_bytes, error) ||
        TrimAscii(digest_bytes) != catalog_digest) {
        error = "metric schema digest file does not match the catalog";
        return false;
    }
    const fs::path manifest_path = catalog_path.parent_path().parent_path() /
                                   "manifest.json";
    std::string manifest_bytes;
    if (!ReadFile(manifest_path, manifest_bytes, error)) return false;
    google::protobuf::Struct manifest;
    if (!google::protobuf::util::JsonStringToMessage(
             manifest_bytes, &manifest).ok()) {
        error = "contract snapshot manifest is invalid";
        return false;
    }
    const auto* source_digest = JsonStruct(manifest, "source_digest");
    const auto* artifact_digest = JsonStruct(manifest, "artifact_digest");
    const auto* metric_schemas = JsonStruct(manifest, "metric_schemas");
    const auto* schema_metadata = metric_schemas
        ? JsonStruct(*metric_schemas, "maze.metrics.v2") : nullptr;
    const auto* canonical_digest = schema_metadata
        ? JsonStruct(*schema_metadata, "canonical_digest") : nullptr;
    const auto* files = JsonStruct(manifest, "files");
    const std::string catalog_relative = "schemas/maze.metrics.v2.json";
    const std::string digest_relative = "schemas/maze.metrics.v2.sha256";
    if (!JsonStringEquals(manifest, "package", contract.package_name) ||
        !JsonStringEquals(manifest, "version", contract.package_version) ||
        !JsonStringEquals(manifest, "platform", contract.platform) ||
        !JsonStringEquals(manifest, "generator_identity",
                          contract.generator_identity) ||
        !source_digest ||
        !JsonStringEquals(*source_digest, "algorithm", "sha256") ||
        !JsonStringEquals(*source_digest, "hex",
                          contract.source_digest.hex) ||
        !artifact_digest ||
        !JsonStringEquals(*artifact_digest, "algorithm", "sha256") ||
        !JsonStringEquals(*artifact_digest, "hex",
                          contract.artifact_digest.hex) ||
        !files ||
        CanonicalFileTableDigest(*files) !=
            contract.artifact_digest.hex ||
        !schema_metadata ||
        !JsonNumberEquals(*schema_metadata, "schema_version", 2.0) ||
        !JsonStringEquals(*schema_metadata, "path", catalog_relative) ||
        !JsonStringEquals(*schema_metadata, "digest_path", digest_relative) ||
        !canonical_digest ||
        !JsonStringEquals(*canonical_digest, "algorithm", "sha256") ||
        !JsonStringEquals(*canonical_digest, "hex", catalog_digest) ||
        !JsonStringEquals(*files, catalog_relative, catalog_digest) ||
        !JsonStringEquals(*files, digest_relative,
                          Sha256Hex(digest_bytes))) {
        error = "metric schema catalog is not bound to the selected contract manifest";
        return false;
    }
    metrics.event_schema_catalog_path = catalog_path.string();
    metrics.event_schema.schema_id = schema_id->second.string_value();
    metrics.event_schema.schema_version = static_cast<uint32_t>(
        schema_version->second.number_value());
    metrics.event_schema.canonical_digest = {"sha256", catalog_digest};
    return true;
}

}  // namespace

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

static std::string GetEnvValue(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : "";
}

static int EnvInt(const char* name, int def) {
    return SafeInt(GetEnvValue(name), def);
}

static bool IsLowerSha256(const std::string& value) {
    if (value.size() != 64) return false;
    for (const char character : value) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
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

    const std::pair<const char*, const char*> required[] = {
        {"contract", "package_name"},
        {"contract", "package_version"},
        {"contract", "source_digest"},
        {"contract", "artifact_digest"},
        {"contract", "platform"},
        {"contract", "generator_identity"},
        {"training_semantics", "training_contract_id"},
        {"training_semantics", "observation_schema_id"},
        {"training_semantics", "observation_schema_version"},
        {"training_semantics", "observation_schema_digest"},
        {"training_semantics", "action_schema_id"},
        {"training_semantics", "action_schema_version"},
        {"training_semantics", "action_schema_digest"},
        {"training_semantics", "reward_schema_id"},
        {"training_semantics", "reward_schema_version"},
        {"training_semantics", "reward_schema_digest"},
        {"training_semantics", "policy_distribution_schema_id"},
        {"training_semantics", "model_architecture_id"},
        {"training_semantics", "semantics_digest"},
        {"policy", "distribution_schema_id"},
        {"policy", "training_temperature"},
        {"policy", "policy_spec_digest"},
        {"policy", "sampling_seed"},
        {"observation", "ray_max_range"},
        {"metrics", "event_schema_catalog"},
        {"reward", "goal_reward"},
        {"reward", "timeout_penalty"},
        {"reward", "progress_budget"},
        {"reward", "stage_8x_first_visit_budget"},
        {"reward", "stage_4x_first_visit_budget"},
        {"reward", "stage_2x_first_visit_budget"},
        {"reward", "wasted_action_penalty"},
        {"task", "task_contract_id"},
        {"task", "task_id"},
        {"task", "task_revision"},
        {"task", "task_config_digest"},
        {"task", "agent_num"},
        {"task", "fixed_map_id"},
        {"task", "fixed_map_checksum_sha256"},
        {"task", "action_rule_id"},
        {"task", "shortest_action_steps"},
        {"model", "expected_obs_dim"},
        {"model", "expected_action_dim"},
        {"model", "model_architecture_id"},
        {"model", "tensor_dtype"},
        {"model", "expected_model_lineage_id"},
    };
    for (const auto& field : required) {
        if (FindValue(entries, field.first, field.second).empty()) {
            LOG_ERROR("Config", "缺少关键配置: %s.%s",
                      field.first, field.second);
            return false;
        }
    }

    // Automatic model evaluation and quality gating are not part of either
    // local or Infra-managed training. Reject stale configurations instead of
    // silently reviving the retired in-training evaluation controller.
    const char* retired_training_controller_fields[] = {
        "stage_8x_sample_budget",
        "stage_4x_sample_budget",
        "stage_2x_sample_budget",
        "evaluation_interval_samples",
        "evaluation_episodes_per_round",
        "stage_8x_success_threshold",
        "stage_4x_success_threshold",
        "stage_2x_success_threshold",
        "final_path_ratio_median_limit",
        "final_path_ratio_p95_limit",
    };
    for (const char* field : retired_training_controller_fields) {
        if (!FindValue(entries, "curriculum", field).empty()) {
            LOG_ERROR(
                "Config",
                "训练配置禁止已退役的课程/评测控制字段: curriculum.%s",
                field);
            return false;
        }
    }
    if (!FindValue(entries, "task", "training_sample_budget").empty()) {
        LOG_ERROR(
            "Config",
            "训练配置禁止样本硬上限: task.training_sample_budget");
        return false;
    }


    const std::pair<const char*, const char*> integer_fields[] = {
        {"training_semantics", "observation_schema_version"},
        {"training_semantics", "action_schema_version"},
        {"training_semantics", "reward_schema_version"},
        {"policy", "sampling_seed"},
        {"observation", "ray_max_range"},
        {"server", "listen_port"},
        {"server", "max_agents"},
        {"server", "run_mode"},
        {"strategy", "grid_size"},
        {"strategy", "replan_interval"},
        {"model", "startup_timeout_ms"},
        {"model", "expected_obs_dim"},
        {"model", "expected_action_dim"},
        {"task", "task_revision"},
        {"task", "agent_num"},
        {"task", "shortest_action_steps"},
        {"model_distribution", "port"},
        {"model_distribution", "poll_interval_ms"},
        {"model_distribution", "rpc_timeout_ms"},
        {"sample_output", "port"},
        {"sample_output", "fragment_samples"},
        {"sample_output", "rpc_timeout_ms"},
        {"sample_output", "max_attempts"},
        {"sample_output", "enqueue_timeout_ms"},
        {"sample_output", "drain_timeout_ms"},
        {"sample_output", "health_timeout_ms"},
        {"sample_output", "status_poll_interval_ms"},
        {"sample_output", "outbound_max_fragments"},
        {"sample_output", "outbound_max_estimated_bytes"},
        {"metrics", "episode_window"},
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
    const std::pair<const char*, const char*> finite_fields[] = {
        {"policy", "training_temperature"},
        {"reward", "goal_reward"},
        {"reward", "timeout_penalty"},
        {"reward", "progress_budget"},
        {"reward", "stage_8x_first_visit_budget"},
        {"reward", "stage_4x_first_visit_budget"},
        {"reward", "stage_2x_first_visit_budget"},
        {"reward", "wasted_action_penalty"},
    };
    for (const auto& field : finite_fields) {
        const std::string value = FindValue(entries, field.first, field.second);
        if (!value.empty() && !std::isfinite(SafeDouble(value, NAN))) {
            LOG_ERROR("Config", "浮点配置无效: %s.%s",
                      field.first, field.second);
            return false;
        }
    }
    const std::string sample_output_enabled =
        FindValue(entries, "sample_output", "enabled");
    if (!sample_output_enabled.empty() &&
        !IsStrictBool(sample_output_enabled)) {
        LOG_ERROR("Config", "布尔配置无效: sample_output.enabled");
        return false;
    }

    out_config.contract.package_name =
        FindValue(entries, "contract", "package_name");
    out_config.contract.package_version =
        FindValue(entries, "contract", "package_version");
    out_config.contract.source_digest.hex =
        FindValue(entries, "contract", "source_digest");
    out_config.contract.artifact_digest.hex =
        FindValue(entries, "contract", "artifact_digest");
    out_config.contract.platform =
        FindValue(entries, "contract", "platform");
    out_config.contract.generator_identity =
        FindValue(entries, "contract", "generator_identity");

    auto& semantics = out_config.training_semantics;
    semantics.training_contract_id =
        FindValue(entries, "training_semantics", "training_contract_id");
    semantics.observation_schema.schema_id =
        FindValue(entries, "training_semantics", "observation_schema_id");
    semantics.observation_schema.schema_version = static_cast<uint32_t>(
        SafeSize(FindValue(entries, "training_semantics",
                           "observation_schema_version"), 0));
    semantics.observation_schema.canonical_digest.hex =
        FindValue(entries, "training_semantics",
                  "observation_schema_digest");
    semantics.action_schema.schema_id =
        FindValue(entries, "training_semantics", "action_schema_id");
    semantics.action_schema.schema_version = static_cast<uint32_t>(
        SafeSize(FindValue(entries, "training_semantics",
                           "action_schema_version"), 0));
    semantics.action_schema.canonical_digest.hex =
        FindValue(entries, "training_semantics", "action_schema_digest");
    semantics.reward_schema.schema_id =
        FindValue(entries, "training_semantics", "reward_schema_id");
    semantics.reward_schema.schema_version = static_cast<uint32_t>(
        SafeSize(FindValue(entries, "training_semantics",
                           "reward_schema_version"), 0));
    semantics.reward_schema.canonical_digest.hex =
        FindValue(entries, "training_semantics", "reward_schema_digest");
    semantics.policy_distribution_schema_id =
        FindValue(entries, "training_semantics",
                  "policy_distribution_schema_id");
    semantics.model_architecture_id =
        FindValue(entries, "training_semantics", "model_architecture_id");
    semantics.semantics_digest.hex =
        FindValue(entries, "training_semantics", "semantics_digest");

    out_config.policy.distribution_schema_id =
        FindValue(entries, "policy", "distribution_schema_id");
    out_config.policy.training_temperature = SafeDouble(
        FindValue(entries, "policy", "training_temperature"), 0.0);
    out_config.policy.policy_spec_digest.hex =
        FindValue(entries, "policy", "policy_spec_digest");
    out_config.policy.sampling_seed = static_cast<uint32_t>(SafeSize(
        FindValue(entries, "policy", "sampling_seed"), 0));
    out_config.observation.ray_max_range = SafeInt(
        FindValue(entries, "observation", "ray_max_range"), 0);

    out_config.reward.goal_reward = static_cast<float>(SafeDouble(
        FindValue(entries, "reward", "goal_reward"), 0.0));
    out_config.reward.timeout_penalty = static_cast<float>(SafeDouble(
        FindValue(entries, "reward", "timeout_penalty"), 0.0));
    out_config.reward.progress_budget = static_cast<float>(SafeDouble(
        FindValue(entries, "reward", "progress_budget"), -1.0));
    out_config.reward.stage_8x_first_visit_budget = static_cast<float>(
        SafeDouble(FindValue(entries, "reward",
                             "stage_8x_first_visit_budget"), -1.0));
    out_config.reward.stage_4x_first_visit_budget = static_cast<float>(
        SafeDouble(FindValue(entries, "reward",
                             "stage_4x_first_visit_budget"), -1.0));
    out_config.reward.stage_2x_first_visit_budget = static_cast<float>(
        SafeDouble(FindValue(entries, "reward",
                             "stage_2x_first_visit_budget"), -1.0));
    out_config.reward.wasted_action_penalty = static_cast<float>(
        SafeDouble(FindValue(entries, "reward",
                             "wasted_action_penalty"), 0.0));

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

    // --- strategy ---
    out_config.strategy.grid_size        = SafeInt(FindValue(entries, "strategy", "grid_size"),        500);
    out_config.strategy.replan_interval  = SafeInt(FindValue(entries, "strategy", "replan_interval"),  10);

    // --- model ---
    std::string evaluation_dir =
        FindValue(entries, "model", "evaluation_dir");
    if (!evaluation_dir.empty()) {
        out_config.model.evaluation_dir = evaluation_dir;
    }
    std::string local_train_dir =
        FindValue(entries, "model", "local_train_dir");
    if (!local_train_dir.empty()) {
        out_config.model.local_train_dir = local_train_dir;
    }
    std::string local_test_dir =
        FindValue(entries, "model", "local_test_dir");
    if (!local_test_dir.empty()) {
        out_config.model.local_test_dir = local_test_dir;
    }
    std::string manifest_name = FindValue(entries, "model", "manifest_name");
    if (!manifest_name.empty()) {
        out_config.model.manifest_name = manifest_name;
    }
    out_config.model.startup_timeout_ms =
        SafeInt(FindValue(entries, "model", "startup_timeout_ms"), 30000);
    out_config.model.expected_obs_dim =
        SafeInt(FindValue(entries, "model", "expected_obs_dim"), 17);
    out_config.model.expected_action_dim =
        SafeInt(FindValue(entries, "model", "expected_action_dim"), 9);
    out_config.model.observation_schema_id =
        semantics.observation_schema.schema_id;
    out_config.model.action_schema_id = semantics.action_schema.schema_id;
    out_config.model.model_architecture_id =
        FindValue(entries, "model", "model_architecture_id");
    out_config.model.tensor_dtype =
        FindValue(entries, "model", "tensor_dtype");
    out_config.model.expected_model_lineage_id =
        FindValue(entries, "model", "expected_model_lineage_id");

    // --- task ---
    out_config.task.task_contract_id =
        FindValue(entries, "task", "task_contract_id");
    std::string task_id = FindValue(entries, "task", "task_id");
    if (!task_id.empty()) out_config.task.task_id = task_id;
    out_config.task.task_revision = static_cast<uint64_t>(SafeSize(
        FindValue(entries, "task", "task_revision"), 1));
    out_config.task.task_config_digest.hex =
        FindValue(entries, "task", "task_config_digest");
    out_config.task.agent_num = SafeInt(
        FindValue(entries, "task", "agent_num"), 4);
    std::string fixed_map_id =
        FindValue(entries, "task", "fixed_map_id");
    if (!fixed_map_id.empty()) out_config.task.fixed_map_id = fixed_map_id;
    std::string fixed_map_checksum =
        FindValue(entries, "task", "fixed_map_checksum_sha256");
    if (!fixed_map_checksum.empty()) {
        out_config.task.fixed_map_checksum_sha256 = fixed_map_checksum;
    }
    out_config.task.action_rule_id =
        FindValue(entries, "task", "action_rule_id");
    out_config.task.shortest_action_steps = SafeInt(
        FindValue(entries, "task", "shortest_action_steps"), 0);

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
    out_config.sample_output.status_poll_interval_ms =
        SafeInt(FindValue(entries, "sample_output", "status_poll_interval_ms"),
                200);
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

    std::string listen_port = GetEnvValue("RL_AISERVER_LISTEN_PORT");
    if (!listen_port.empty()) {
        out_config.server.listen_port = SafeInt(listen_port, out_config.server.listen_port);
    }
    std::string run_mode = GetEnvValue("RL_AISERVER_RUN_MODE");
    if (!run_mode.empty()) {
        const int candidate = aiserver_mode::Parse(run_mode);
        if (aiserver_mode::IsValid(candidate)) {
            out_config.server.run_mode = candidate;
        } else {
            LOG_ERROR("Config", "环境运行模式无效: %s",
                      run_mode.c_str());
            return false;
        }
    }
    std::string model_distributor_host =
        GetEnvValue("RL_MODEL_DISTRIBUTOR_HOST");
    if (!model_distributor_host.empty()) {
        out_config.model_distribution.host = model_distributor_host;
    }
    std::string model_distributor_port =
        GetEnvValue("RL_MODEL_DISTRIBUTOR_PORT");
    if (!model_distributor_port.empty()) {
        out_config.model_distribution.port = SafeInt(
            model_distributor_port, out_config.model_distribution.port);
    }
    std::string local_train_root =
        GetEnvValue("RL_LOCAL_TRAIN_ROOT");
    if (!local_train_root.empty()) {
        out_config.model.local_train_dir = local_train_root;
    }
    std::string evaluation_model_dir =
        GetEnvValue("RL_EVALUATION_MODEL_DIR");
    if (!evaluation_model_dir.empty()) {
        out_config.model.evaluation_dir = evaluation_model_dir;
    }
    std::string local_test_model_dir =
        GetEnvValue("RL_LOCAL_TEST_MODEL_DIR");
    if (!local_test_model_dir.empty()) {
        out_config.model.local_test_dir = local_test_model_dir;
    }
    std::string sd_host = GetEnvValue("RL_SAMPLE_DISTRIBUTOR_HOST");
    if (!sd_host.empty()) {
        out_config.sample_output.host = sd_host;
    }
    std::string sd_port = GetEnvValue("RL_SAMPLE_DISTRIBUTOR_PORT");
    if (!sd_port.empty()) {
        out_config.sample_output.port = SafeInt(sd_port, out_config.sample_output.port);
    }
    std::string env_aiserver_id = GetEnvValue("RL_AISERVER_ID");
    if (!env_aiserver_id.empty()) {
        out_config.sample_output.aiserver_id = env_aiserver_id;
    }
    std::string env_env_id = GetEnvValue("RL_ENVIRONMENT_INSTANCE_ID");
    if (!env_env_id.empty()) {
        out_config.sample_output.env_id = env_env_id;
    }
    out_config.sample_output.fragment_samples =
        EnvInt("RL_SAMPLE_FRAGMENT_SIZE", out_config.sample_output.fragment_samples);
    out_config.sample_output.rpc_timeout_ms =
        EnvInt("RL_SAMPLE_RPC_TIMEOUT_MS", out_config.sample_output.rpc_timeout_ms);
    out_config.sample_output.max_attempts =
        EnvInt("RL_SAMPLE_MAX_ATTEMPTS", out_config.sample_output.max_attempts);
    out_config.sample_output.enqueue_timeout_ms =
        EnvInt("RL_SAMPLE_ENQUEUE_TIMEOUT_MS", out_config.sample_output.enqueue_timeout_ms);
    out_config.sample_output.drain_timeout_ms =
        EnvInt("RL_SAMPLE_DRAIN_TIMEOUT_MS", out_config.sample_output.drain_timeout_ms);
    out_config.sample_output.status_poll_interval_ms =
        EnvInt("RL_SAMPLE_STATUS_POLL_INTERVAL_MS",
               out_config.sample_output.status_poll_interval_ms);
    out_config.model.startup_timeout_ms =
        EnvInt("RL_MODEL_STARTUP_TIMEOUT_MS", out_config.model.startup_timeout_ms);
    out_config.model_distribution.poll_interval_ms =
        EnvInt(
            "RL_MODEL_POLL_INTERVAL_MS",
            out_config.model_distribution.poll_interval_ms);
    if (!GetEnvValue("RL_TRAINING_SAMPLE_BUDGET").empty()) {
        LOG_ERROR(
            "Config",
            "RL_TRAINING_SAMPLE_BUDGET was retired; training has no hard cap");
        return false;
    }
    out_config.metrics.episode_window = SafeSize(
        GetEnvValue("RL_EPISODE_METRICS_WINDOW"),
        out_config.metrics.episode_window);
    if (out_config.metrics.episode_window == 0) {
        out_config.metrics.episode_window = 100;
    }
    std::string metric_schema_error;
    if (!LoadMetricEventSchema(
            yaml_path,
            FindValue(entries, "metrics", "event_schema_catalog"),
            out_config.contract,
            out_config.metrics, metric_schema_error)) {
        LOG_ERROR("Config", "%s", metric_schema_error.c_str());
        return false;
    }

    const auto digest_valid = [](const DigestConfig& digest) {
        return digest.algorithm == "sha256" && IsLowerSha256(digest.hex);
    };
    const bool immutable_identity_valid =
        out_config.contract.package_name == "rl-contracts" &&
        out_config.contract.package_version == "0.11.0" &&
        out_config.contract.source_digest.hex ==
            "7e3eb7227e67a2a880130c9c82f87041691c0095f838a60f80abc1f387c1c5b3" &&
        out_config.contract.artifact_digest.hex ==
            "077ac6d61486fafd5f0430eeb05a492764b36e073282f6d7626d0414bb5b2ddf" &&
        out_config.contract.platform == "linux/arm64" &&
        out_config.contract.generator_identity ==
            "0eb73fc2cb675bdb34bf3db9c99dae62a82f93a5e3a72db84dcf3936464729c8" &&
        out_config.training_semantics.training_contract_id ==
            "maze.training.v3" &&
        out_config.training_semantics.observation_schema.schema_id ==
            "maze.observation.v3" &&
        out_config.training_semantics.observation_schema.schema_version == 1 &&
        out_config.training_semantics.observation_schema.canonical_digest.hex ==
            "7cee41136020f3ffc8c6ae799f630d55d0588c6a99ab7f717eac3b3d08aa18b4" &&
        out_config.training_semantics.action_schema.schema_id ==
            "maze.action.v1" &&
        out_config.training_semantics.action_schema.schema_version == 1 &&
        out_config.training_semantics.action_schema.canonical_digest.hex ==
            "ce84c564e128f98adcc48fd420ac0df5acea61774a25de8705b602464009cfd8" &&
        out_config.training_semantics.reward_schema.schema_id ==
            "maze.reward.v4" &&
        out_config.training_semantics.reward_schema.schema_version == 1 &&
        out_config.training_semantics.reward_schema.canonical_digest.hex ==
            "ed284084b79413473d5053b6d3f69320d2a4639c81451ba598ca45ac8ce15929" &&
        out_config.training_semantics.policy_distribution_schema_id ==
            "categorical.logits.v1" &&
        out_config.training_semantics.model_architecture_id ==
            "maze.mlp-17x64x64.v1" &&
        out_config.training_semantics.semantics_digest.hex ==
            "6cd834542f8263135b4bfd069f372ddfdb99334060d305f58b00ce56eea10b4c" &&
        out_config.policy.distribution_schema_id ==
            "categorical.logits.v1" &&
        out_config.policy.policy_spec_digest.hex ==
            "e1efb81040681fd13fdae439caab09778c8e0b0f6bba16b7808c30fbbb632617";
    if (!immutable_identity_valid ||
        !digest_valid(out_config.contract.source_digest) ||
        !digest_valid(out_config.contract.artifact_digest) ||
        !digest_valid(out_config.training_semantics.observation_schema.canonical_digest) ||
        !digest_valid(out_config.training_semantics.action_schema.canonical_digest) ||
        !digest_valid(out_config.training_semantics.reward_schema.canonical_digest) ||
        out_config.metrics.event_schema.schema_id != "maze.metrics.v2" ||
        out_config.metrics.event_schema.schema_version != 2 ||
        !digest_valid(out_config.metrics.event_schema.canonical_digest) ||
        !digest_valid(out_config.training_semantics.semantics_digest) ||
        !digest_valid(out_config.policy.policy_spec_digest)) {
        LOG_ERROR("Config", "0.11.0 contract/training identity mismatch");
        return false;
    }
    if (out_config.task.task_contract_id != "maze.task.v3" ||
        out_config.task.task_id != "maze.fixed.single-map.v1" ||
        out_config.task.task_revision != 2 ||
        !digest_valid(out_config.task.task_config_digest) ||
        out_config.task.task_config_digest.hex !=
            "f16411393f778b7a2ffaf688e80f138dc33bd0709f47190c3f7b0f5946178b5b" ||
        out_config.task.agent_num != 4 ||
        out_config.task.fixed_map_id != "maze_117436372" ||
        out_config.task.fixed_map_checksum_sha256 !=
            "861e0bb22a8b9a2ed689527d080c65ec2c822367e985c49753e1be9cf3ca8ae9" ||
        out_config.task.action_rule_id !=
            "maze.action.9-way.no-corner-cut.v1" ||
        out_config.task.shortest_action_steps != 188) {
        LOG_ERROR("Config", "fixed Maze task identity mismatch");
        return false;
    }
    if (out_config.model.expected_obs_dim != 17 ||
        out_config.model.expected_action_dim != 9 ||
        out_config.model.model_architecture_id !=
            out_config.training_semantics.model_architecture_id ||
        out_config.model.tensor_dtype != "float32" ||
        out_config.model.expected_model_lineage_id.empty() ||
        out_config.policy.training_temperature != 1.0 ||
        out_config.observation.ray_max_range <= 0 ||
        std::fabs(out_config.reward.goal_reward - 10.0f) > 1e-6f ||
        std::fabs(out_config.reward.timeout_penalty + 2.0f) > 1e-6f ||
        std::fabs(out_config.reward.progress_budget - 1.0f) > 1e-6f ||
        std::fabs(out_config.reward.stage_8x_first_visit_budget - 0.75f) > 1e-6f ||
        std::fabs(out_config.reward.stage_4x_first_visit_budget - 0.25f) > 1e-6f ||
        std::fabs(out_config.reward.stage_2x_first_visit_budget) > 1e-6f ||
        std::fabs(out_config.reward.wasted_action_penalty + 0.002f) > 1e-6f ||
        out_config.reward.timeout_penalty +
                out_config.reward.progress_budget +
                out_config.reward.stage_8x_first_visit_budget >=
            0.0f) {
        LOG_ERROR("Config", "model, policy, observation or Reward V4 mismatch");
        return false;
    }
    const bool runtime_values_valid =
        out_config.server.listen_port > 0 &&
        out_config.server.listen_port <= 65535 &&
        out_config.server.max_agents >= out_config.task.agent_num &&
        aiserver_mode::IsValid(out_config.server.run_mode) &&
        out_config.strategy.grid_size > 0 &&
        out_config.strategy.replan_interval >= 0 &&
        out_config.model.startup_timeout_ms > 0 &&
        out_config.model_distribution.port > 0 &&
        out_config.model_distribution.port <= 65535 &&
        out_config.model_distribution.poll_interval_ms > 0 &&
        out_config.model_distribution.rpc_timeout_ms > 0 &&
        out_config.sample_output.port > 0 &&
        out_config.sample_output.port <= 65535 &&
        out_config.sample_output.fragment_samples > 0 &&
        out_config.sample_output.rpc_timeout_ms > 0 &&
        out_config.sample_output.max_attempts > 0 &&
        out_config.sample_output.enqueue_timeout_ms > 0 &&
        out_config.sample_output.drain_timeout_ms > 0 &&
        out_config.sample_output.health_timeout_ms > 0 &&
        out_config.sample_output.status_poll_interval_ms > 0 &&
        out_config.sample_output.outbound_max_fragments >=
            static_cast<std::size_t>(out_config.task.agent_num) &&
        out_config.sample_output.outbound_max_estimated_bytes > 0 &&
        out_config.metrics.episode_window > 0 &&
        out_config.metrics.episode_window <= 1000000;
    if (!runtime_values_valid) {
        LOG_ERROR("Config", "运行时数值配置无效");
        return false;
    }
    const int64_t fragment_quantum =
        static_cast<int64_t>(out_config.task.agent_num) *
        static_cast<int64_t>(out_config.sample_output.fragment_samples);
    if (fragment_quantum != 512 || out_config.server.max_agents < 4 ||
        out_config.model_distribution.contract_version != "0.11.0") {
        LOG_ERROR("Config", "sample quantum or runtime contract mismatch");
        return false;
    }

    LOG_INFO("Config", "server: port=%d, max_agents=%d, run_mode=%d(%s)",
             out_config.server.listen_port, out_config.server.max_agents,
             out_config.server.run_mode,
             aiserver_mode::Workload(out_config.server.run_mode));
    LOG_INFO("Config", "strategy: grid=%d, replan=%d",
             out_config.strategy.grid_size,
             out_config.strategy.replan_interval);
    LOG_INFO("Config", "model: evaluation_dir=%s, evaluation_manifest=%s, local_train=%s, local_test_dir=%s, manifest=%s, startup_timeout_ms=%d, shape=[%d]->[%d], schemas=%s/%s",
             out_config.model.evaluation_dir.c_str(),
             LocalEvaluationManifestPath(out_config.model).c_str(),
             out_config.model.local_train_dir.c_str(),
             out_config.model.local_test_dir.c_str(),
             out_config.model.manifest_name.c_str(),
             out_config.model.startup_timeout_ms,
             out_config.model.expected_obs_dim,
             out_config.model.expected_action_dim,
             out_config.model.observation_schema_id.c_str(),
             out_config.model.action_schema_id.c_str());
    LOG_INFO("Config", "task: id=%s revision=%llu agents=%d map=%s",
             out_config.task.task_id.c_str(),
             static_cast<unsigned long long>(out_config.task.task_revision),
             out_config.task.agent_num,
             out_config.task.fixed_map_id.c_str());
    LOG_INFO("Config", "model_distribution: target=%s:%d, poll_interval_ms=%d, rpc_timeout_ms=%d, contract=%s",
             out_config.model_distribution.host.c_str(),
             out_config.model_distribution.port,
             out_config.model_distribution.poll_interval_ms,
             out_config.model_distribution.rpc_timeout_ms,
             out_config.model_distribution.contract_version.c_str());
    const bool sample_output_active =
        out_config.server.run_mode == aiserver_mode::kTraining &&
        out_config.sample_output.enabled;
    LOG_INFO("Config", "sample_output: configured_enabled=%s, active=%s, target=%s:%d, fragment=%d, rpc_timeout_ms=%d, status_poll_interval_ms=%d, attempts=%d, queue=%zu/%zuB, aiserver_id=%s, env_id=%s",
             out_config.sample_output.enabled ? "true" : "false",
             sample_output_active ? "true" : "false",
             out_config.sample_output.host.c_str(),
             out_config.sample_output.port,
             out_config.sample_output.fragment_samples,
             out_config.sample_output.rpc_timeout_ms,
             out_config.sample_output.status_poll_interval_ms,
             out_config.sample_output.max_attempts,
             out_config.sample_output.outbound_max_fragments,
             out_config.sample_output.outbound_max_estimated_bytes,
             out_config.sample_output.aiserver_id.c_str(),
             out_config.sample_output.env_id.c_str());
    LOG_INFO("Config", "metrics: episode_window=%zu",
             out_config.metrics.episode_window);
    return true;
}
