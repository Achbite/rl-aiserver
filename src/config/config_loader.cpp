#include "config/config_loader.h"

#include "ai/maze_reward.h"
#include "log/logger.h"

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <regex>
#include <set>
#include <sstream>
#include <vector>
#include <cctype>
#include <cmath>
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

bool JsonStringValue(const google::protobuf::Struct& document,
                     const std::string& name,
                     std::string& output) {
    const auto* value = JsonField(document, name);
    if (!value || value->kind_case() !=
                      google::protobuf::Value::kStringValue ||
        value->string_value().empty()) {
        return false;
    }
    output = value->string_value();
    return true;
}

bool JsonPositiveInteger(const google::protobuf::Struct& document,
                         const std::string& name,
                         int& output) {
    const auto* value = JsonField(document, name);
    if (!value || value->kind_case() !=
                      google::protobuf::Value::kNumberValue ||
        !std::isfinite(value->number_value()) ||
        value->number_value() < 1.0 ||
        value->number_value() >
            static_cast<double>(std::numeric_limits<int>::max()) ||
        std::floor(value->number_value()) != value->number_value()) {
        return false;
    }
    output = static_cast<int>(value->number_value());
    return true;
}

bool JsonHasExactFields(const google::protobuf::Struct& document,
                        std::initializer_list<const char*> expected) {
    if (document.fields_size() != static_cast<int>(expected.size())) {
        return false;
    }
    for (const char* name : expected) {
        if (document.fields().count(name) != 1) return false;
    }
    return true;
}

bool HasVersionSuffix(const std::string& value) {
    static const std::regex suffix("\\.v[0-9]+$");
    return std::regex_search(value, suffix);
}

bool IsSha256Hex(const std::string& value) {
    if (value.size() != 64) return false;
    return std::all_of(value.begin(), value.end(), [](char character) {
        return (character >= '0' && character <= '9') ||
               (character >= 'a' && character <= 'f');
    });
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
        schema_id->second.string_value() != "maze.metrics" ||
        schema_version->second.number_value() != 1.0) {
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
        ? JsonStruct(*metric_schemas, "maze.metrics") : nullptr;
    const auto* canonical_digest = schema_metadata
        ? JsonStruct(*schema_metadata, "canonical_digest") : nullptr;
    const auto* files = JsonStruct(manifest, "files");
    const std::string catalog_relative = "schemas/maze.metrics.json";
    const std::string digest_relative = "schemas/maze.metrics.sha256";
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
        !JsonNumberEquals(*schema_metadata, "schema_version", 1.0) ||
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

bool ReadTrainingSchema(const google::protobuf::Struct& document,
                        const std::string& expected_id,
                        SchemaConfig& schema) {
    int version = 0;
    if (!JsonHasExactFields(
            document, {"canonical_digest", "schema_id", "schema_version"}) ||
        !JsonStringValue(document, "schema_id", schema.schema_id) ||
        schema.schema_id != expected_id || HasVersionSuffix(schema.schema_id) ||
        !JsonPositiveInteger(document, "schema_version", version) ||
        !JsonStringValue(document, "canonical_digest",
                         schema.canonical_digest.hex) ||
        !IsSha256Hex(schema.canonical_digest.hex)) {
        return false;
    }
    schema.schema_version = static_cast<uint32_t>(version);
    schema.canonical_digest.algorithm = "sha256";
    return true;
}

bool LoadTrainingContract(const std::string& yaml_path,
                          const std::string& configured_path,
                          ContractConfig& contract,
                          TrainingContractConfig& training,
                          PolicyConfig& policy,
                          ModelConfig& model,
                          std::string& error) {
    namespace fs = std::filesystem;
    fs::path descriptor_path(configured_path);
    if (descriptor_path.is_relative()) {
        descriptor_path = fs::path(yaml_path).parent_path() / descriptor_path;
    }
    std::error_code fs_error;
    descriptor_path = fs::weakly_canonical(descriptor_path, fs_error);
    if (fs_error || !fs::is_regular_file(descriptor_path, fs_error) ||
        fs::is_symlink(fs::symlink_status(descriptor_path, fs_error)) ||
        descriptor_path.filename() != "training-contract.json") {
        error = "training contract descriptor is missing or invalid: " +
                descriptor_path.string();
        return false;
    }
    std::string bytes;
    if (!ReadFile(descriptor_path, bytes, error)) return false;
    const std::string descriptor_digest = Sha256Hex(bytes);
    if (!IsSha256Hex(descriptor_digest)) {
        error = "cannot calculate training contract digest";
        return false;
    }
    fs::path digest_path = descriptor_path;
    digest_path.replace_extension(".sha256");
    std::string digest_bytes;
    if (!ReadFile(digest_path, digest_bytes, error) ||
        TrimAscii(digest_bytes) != descriptor_digest) {
        error = "training contract digest file does not match the descriptor";
        return false;
    }

    const fs::path manifest_path = descriptor_path.parent_path().parent_path() /
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
    const auto* descriptor_metadata = JsonStruct(manifest, "training_contract");
    const auto* canonical_digest = descriptor_metadata
        ? JsonStruct(*descriptor_metadata, "canonical_digest") : nullptr;
    const auto* files = JsonStruct(manifest, "files");
    constexpr const char* kDescriptorRelative =
        "schemas/training-contract.json";
    constexpr const char* kDigestRelative =
        "schemas/training-contract.sha256";
    if (!JsonStringEquals(manifest, "package", contract.package_name) ||
        !JsonStringEquals(manifest, "version", contract.package_version) ||
        !JsonStringEquals(manifest, "platform", contract.platform) ||
        !JsonStringEquals(manifest, "generator_identity",
                          contract.generator_identity) ||
        !source_digest ||
        !JsonStringEquals(*source_digest, "algorithm", "sha256") ||
        !JsonStringEquals(*source_digest, "hex", contract.source_digest.hex) ||
        !artifact_digest ||
        !JsonStringEquals(*artifact_digest, "algorithm", "sha256") ||
        !JsonStringEquals(*artifact_digest, "hex",
                          contract.artifact_digest.hex) ||
        !files || CanonicalFileTableDigest(*files) !=
                      contract.artifact_digest.hex ||
        !descriptor_metadata ||
        !JsonStringEquals(*descriptor_metadata, "path", kDescriptorRelative) ||
        !JsonStringEquals(*descriptor_metadata, "digest_path", kDigestRelative) ||
        !canonical_digest ||
        !JsonStringEquals(*canonical_digest, "algorithm", "sha256") ||
        !JsonStringEquals(*canonical_digest, "hex", descriptor_digest) ||
        !JsonStringEquals(*files, kDescriptorRelative, descriptor_digest) ||
        !JsonStringEquals(*files, kDigestRelative, Sha256Hex(digest_bytes))) {
        error = "training contract is not bound to the selected artifact";
        return false;
    }

    google::protobuf::Struct document;
    if (!google::protobuf::util::JsonStringToMessage(bytes, &document).ok() ||
        !JsonHasExactFields(
            document,
            {"action_count", "action_schema", "hidden_dimension",
             "model_architecture_id", "observation_dimension",
             "observation_schema", "policy", "reward_schema", "rollout",
             "tensor_dtype", "training_contract_id"})) {
        error = "training contract JSON shape is invalid";
        return false;
    }
    const auto* observation_schema = JsonStruct(document, "observation_schema");
    const auto* action_schema = JsonStruct(document, "action_schema");
    const auto* reward_schema = JsonStruct(document, "reward_schema");
    const auto* policy_document = JsonStruct(document, "policy");
    const auto* rollout = JsonStruct(document, "rollout");
    std::string distribution_schema_id;
    std::string sampling;
    if (!observation_schema || !action_schema || !reward_schema ||
        !policy_document || !rollout ||
        !ReadTrainingSchema(*observation_schema, "maze.observation",
                            training.observation_schema) ||
        !ReadTrainingSchema(*action_schema, "maze.action",
                            training.action_schema) ||
        !ReadTrainingSchema(*reward_schema, "maze.reward",
                            training.reward_schema) ||
        !JsonStringValue(document, "training_contract_id",
                         training.training_contract_id) ||
        training.training_contract_id != "maze.training" ||
        HasVersionSuffix(training.training_contract_id) ||
        !JsonStringValue(document, "model_architecture_id",
                         training.model_architecture_id) ||
        training.model_architecture_id != "maze.mlp-17x64x64" ||
        HasVersionSuffix(training.model_architecture_id) ||
        !JsonStringValue(document, "tensor_dtype", training.tensor_dtype) ||
        training.tensor_dtype != "float32" ||
        !JsonPositiveInteger(document, "observation_dimension",
                             training.observation_dimension) ||
        training.observation_dimension != 17 ||
        !JsonPositiveInteger(document, "action_count", training.action_count) ||
        training.action_count != 9 ||
        !JsonPositiveInteger(document, "hidden_dimension",
                             training.hidden_dimension) ||
        training.hidden_dimension != 64 ||
        !JsonHasExactFields(
            *policy_document,
            {"distribution_schema_id", "sampling", "temperature"}) ||
        !JsonStringValue(*policy_document, "distribution_schema_id",
                         distribution_schema_id) ||
        distribution_schema_id != "categorical.logits" ||
        HasVersionSuffix(distribution_schema_id) ||
        !JsonStringValue(*policy_document, "sampling", sampling) ||
        sampling != "stochastic" ||
        !JsonNumberEquals(*policy_document, "temperature", 1.0) ||
        !JsonHasExactFields(
            *rollout,
            {"finite_rule_id", "gae_formula_id", "model_pin_semantics_id",
             "numeric_dtype", "terminal_bootstrap_semantics_id",
             "value_head_abi_id", "value_target_formula_id"}) ||
        !JsonStringValue(*rollout, "finite_rule_id",
                         training.finite_rule_id) ||
        !JsonStringValue(*rollout, "gae_formula_id",
                         training.gae_formula_id) ||
        !JsonStringValue(*rollout, "model_pin_semantics_id",
                         training.model_pin_semantics_id) ||
        !JsonStringValue(*rollout, "numeric_dtype",
                         training.numeric_dtype) ||
        !JsonStringValue(*rollout, "terminal_bootstrap_semantics_id",
                         training.terminal_bootstrap_semantics_id) ||
        !JsonStringValue(*rollout, "value_head_abi_id",
                         training.value_head_abi_id) ||
        !JsonStringValue(*rollout, "value_target_formula_id",
                         training.value_target_formula_id)) {
        error = "training contract is unsupported by this AIServer";
        return false;
    }
    for (const std::string* identifier : {
             &training.finite_rule_id, &training.gae_formula_id,
             &training.model_pin_semantics_id,
             &training.terminal_bootstrap_semantics_id,
             &training.value_head_abi_id,
             &training.value_target_formula_id}) {
        if (HasVersionSuffix(*identifier)) {
            error = "training contract contains a versioned ID";
            return false;
        }
    }
    training.canonical_digest = {"sha256", descriptor_digest};
    policy.training_temperature = 1.0;
    model.expected_obs_dim = training.observation_dimension;
    model.expected_action_dim = training.action_count;
    model.observation_schema_id = training.observation_schema.schema_id;
    model.action_schema_id = training.action_schema.schema_id;
    model.model_architecture_id = training.model_architecture_id;
    model.tensor_dtype = training.tensor_dtype;
    contract.training_contract_path = descriptor_path.string();
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

static std::string ComputeTaskConfigDigest(const AIServerConfig& config) {
    std::ostringstream canonical;
    canonical
        << "{\"action_rule_id\":" << JsonQuote(config.task.action_rule_id)
        << ",\"episode_max_steps\":" << config.task.episode_max_steps
        << ",\"fixed_map_checksum_sha256\":"
        << JsonQuote(config.task.fixed_map_checksum_sha256)
        << ",\"fixed_map_id\":" << JsonQuote(config.task.fixed_map_id)
        << ",\"reward\":" << MazeRewardCanonicalParametersJson()
        << ",\"reward_schema_digest\":"
        << JsonQuote(
               config.training_contract.reward_schema.canonical_digest.hex)
        << ",\"reward_schema_id\":"
        << JsonQuote(config.training_contract.reward_schema.schema_id)
        << ",\"shortest_action_steps\":"
        << config.task.shortest_action_steps
        << ",\"task_contract_id\":"
        << JsonQuote(config.task.task_contract_id) << '}';
    return Sha256Hex(canonical.str());
}

// ---- 从 YAML 文件加载配置，依次应用环境和 CLI 覆盖 ----
bool LoadServerConfig(const std::string& yaml_path,
                      const AIServerConfigOverrides& overrides,
                      AIServerConfig& out_config,
                      AIServerConfigLoadReport& report,
                      std::string& error) {
    namespace fs = std::filesystem;
    out_config = AIServerConfig{};
    report = AIServerConfigLoadReport{};
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
        "contract.package_name", "contract.package_version",
        "contract.source_digest", "contract.artifact_digest",
        "contract.platform", "contract.generator_identity",
        "contract.training_contract_path", "policy.sampling_seed",
        "observation.ray_max_range", "strategy.grid_size",
        "strategy.replan_interval", "model.evaluation_model_path",
        "model.local_train_dir", "model.startup_timeout_ms",
        "environment.agent_count", "task.task_contract_id",
        "task.fixed_map_id",
        "task.fixed_map_checksum_sha256", "task.action_rule_id",
        "task.shortest_action_steps", "task.episode_max_steps",
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
        "metrics.event_schema_catalog",
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
        {"contract", "package_name"},
        {"contract", "package_version"},
        {"contract", "source_digest"},
        {"contract", "artifact_digest"},
        {"contract", "platform"},
        {"contract", "generator_identity"},
        {"contract", "training_contract_path"},
        {"policy", "sampling_seed"},
        {"observation", "ray_max_range"},
        {"server", "run_mode"},
        {"server", "listen_port"},
        {"server", "max_agents"},
        {"environment", "agent_count"},
        {"metrics", "event_schema_catalog"},
        {"task", "task_contract_id"},
        {"task", "fixed_map_id"},
        {"task", "fixed_map_checksum_sha256"},
        {"task", "action_rule_id"},
        {"task", "shortest_action_steps"},
        {"task", "episode_max_steps"},
        {"model", "evaluation_model_path"},
        {"model", "local_train_dir"},
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
        {"policy", "sampling_seed"},
        {"observation", "ray_max_range"},
        {"server", "listen_port"},
        {"server", "max_agents"},
        {"environment", "agent_count"},
        {"strategy", "grid_size"},
        {"strategy", "replan_interval"},
        {"model", "startup_timeout_ms"},
        {"task", "shortest_action_steps"},
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
    out_config.contract.training_contract_path =
        FindValue(entries, "contract", "training_contract_path");
    out_config.policy.sampling_seed = static_cast<uint32_t>(SafeSize(
        FindValue(entries, "policy", "sampling_seed"), 0));
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

    // --- strategy ---
    out_config.strategy.grid_size        = SafeInt(FindValue(entries, "strategy", "grid_size"),        500);
    out_config.strategy.replan_interval  = SafeInt(FindValue(entries, "strategy", "replan_interval"),  10);

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

    // --- task ---
    out_config.task.task_contract_id =
        FindValue(entries, "task", "task_contract_id");
    out_config.environment.agent_count = SafeInt(
        FindValue(entries, "environment", "agent_count"), 4);
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
    if (!ReadEnvironment("RL_TASK_MAP_ID", environment_string, error)) {
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }
    if (environment_string.has_value()) {
        out_config.task.fixed_map_id = *environment_string;
        record_environment_override("task.fixed_map_id");
    }
    environment_string.reset();
    if (!ReadEnvironment("RL_TASK_MAP_EXPECTED_SHA256", environment_string,
                         error)) {
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }
    if (environment_string.has_value()) {
        out_config.task.fixed_map_checksum_sha256 = *environment_string;
        record_environment_override("task.fixed_map_checksum_sha256");
    }
    environment_integer.reset();
    if (!ReadEnvironmentInt("RL_TASK_SHORTEST_ACTION_STEPS",
                            environment_integer, error)) {
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }
    if (environment_integer.has_value()) {
        out_config.task.shortest_action_steps = *environment_integer;
        record_environment_override("task.shortest_action_steps");
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

    std::string training_contract_error;
    if (!LoadTrainingContract(
            config_path.string(), out_config.contract.training_contract_path,
            out_config.contract, out_config.training_contract,
            out_config.policy, out_config.model,
            training_contract_error)) {
        error = training_contract_error;
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }

    if (overrides.evaluation_model_path.has_value() &&
        out_config.server.run_mode == aiserver_mode::kTraining) {
        error = "--evaluation-model is invalid for the training workload";
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }
    if (out_config.server.run_mode == aiserver_mode::kEvaluation) {
        const fs::path model_path(out_config.model.evaluation_model_path);
        const auto model_status = fs::symlink_status(model_path, fs_error);
        if (out_config.model.evaluation_model_path.empty() ||
            model_path.filename() != kModelArtifactFile || fs_error ||
            fs::is_symlink(model_status) ||
            !fs::is_regular_file(model_status)) {
            error = "evaluation requires an explicit regular, non-symlink "
                    "SaveModel.onnx";
            LOG_ERROR("Config", "%s: %s", error.c_str(),
                      model_path.c_str());
            return false;
        }
    }

    std::string metric_schema_error;
    if (!LoadMetricEventSchema(
            config_path.string(),
            FindValue(entries, "metrics", "event_schema_catalog"),
            out_config.contract,
            out_config.metrics, metric_schema_error)) {
        LOG_ERROR("Config", "%s", metric_schema_error.c_str());
        return false;
    }

    const auto digest_valid = [](const DigestConfig& digest) {
        return digest.algorithm == "sha256" && IsLowerSha256(digest.hex);
    };
    out_config.task.task_config_digest.hex =
        ComputeTaskConfigDigest(out_config);
    const bool immutable_identity_valid =
        out_config.contract.package_name == "rl-contracts" &&
        out_config.contract.package_version == "0.15.0" &&
        !out_config.contract.platform.empty() &&
        IsLowerSha256(out_config.contract.generator_identity) &&
        out_config.training_contract.training_contract_id ==
            "maze.training" &&
        out_config.training_contract.observation_schema.schema_id ==
            "maze.observation" &&
        out_config.training_contract.action_schema.schema_id ==
            "maze.action" &&
        out_config.training_contract.reward_schema.schema_id ==
            "maze.reward" &&
        out_config.training_contract.model_architecture_id ==
            "maze.mlp-17x64x64";
    if (!immutable_identity_valid ||
        !digest_valid(out_config.contract.source_digest) ||
        !digest_valid(out_config.contract.artifact_digest) ||
        !digest_valid(out_config.training_contract.observation_schema.canonical_digest) ||
        !digest_valid(out_config.training_contract.action_schema.canonical_digest) ||
        !digest_valid(out_config.training_contract.reward_schema.canonical_digest) ||
        out_config.metrics.event_schema.schema_id != "maze.metrics" ||
        out_config.metrics.event_schema.schema_version != 1 ||
        !digest_valid(out_config.metrics.event_schema.canonical_digest) ||
        !digest_valid(out_config.training_contract.canonical_digest)) {
        error = "current contract/training identity mismatch";
        LOG_ERROR("Config", "%s", error.c_str());
        return false;
    }
    if (out_config.task.task_contract_id != "maze.task" ||
        !digest_valid(out_config.task.task_config_digest) ||
        out_config.task.action_rule_id !=
            "maze.action.9-way.no-corner-cut") {
        LOG_ERROR("Config", "Maze task contract identity mismatch");
        return false;
    }
    static const std::regex map_id_pattern("[A-Za-z0-9_-]+");
    if (!std::regex_match(out_config.task.fixed_map_id, map_id_pattern) ||
        !IsLowerSha256(out_config.task.fixed_map_checksum_sha256) ||
        out_config.task.shortest_action_steps <= 0 ||
        out_config.task.episode_max_steps <
            out_config.task.shortest_action_steps) {
        LOG_ERROR("Config", "effective Maze map or episode identity is invalid");
        return false;
    }
    if (out_config.model.expected_obs_dim != 17 ||
        out_config.model.expected_action_dim != 9 ||
        out_config.model.model_architecture_id !=
            out_config.training_contract.model_architecture_id ||
        out_config.model.tensor_dtype != "float32" ||
        out_config.policy.training_temperature != 1.0 ||
        out_config.observation.ray_max_range <= 0) {
        LOG_ERROR("Config", "model, policy or observation contract mismatch");
        return false;
    }
    const bool runtime_values_valid =
        out_config.server.listen_port > 0 &&
        out_config.server.listen_port <= 65535 &&
        out_config.server.max_agents > 0 &&
        out_config.environment.agent_count > 0 &&
        out_config.server.max_agents >= out_config.environment.agent_count &&
        aiserver_mode::IsValid(out_config.server.run_mode) &&
        out_config.strategy.grid_size > 0 &&
        out_config.strategy.replan_interval >= 0 &&
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
    LOG_INFO("Config", "strategy: grid=%d, replan=%d",
             out_config.strategy.grid_size,
             out_config.strategy.replan_interval);
    LOG_INFO("Config", "model: evaluation_model=%s, local_train=%s, startup_timeout_ms=%d, shape=[%d]->[%d], schemas=%s/%s",
             out_config.model.evaluation_model_path.c_str(),
             out_config.model.local_train_dir.c_str(),
             out_config.model.startup_timeout_ms,
             out_config.model.expected_obs_dim,
             out_config.model.expected_action_dim,
             out_config.model.observation_schema_id.c_str(),
             out_config.model.action_schema_id.c_str());
    LOG_INFO("Config", "environment: agent_count=%d; task: contract=%s map=%s digest=%s",
             out_config.environment.agent_count,
             out_config.task.task_contract_id.c_str(),
             out_config.task.fixed_map_id.c_str(),
             out_config.task.task_config_digest.hex.c_str());
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

bool LoadServerConfig(const std::string& yaml_path,
                      AIServerConfig& out_config) {
    AIServerConfigLoadReport report;
    std::string error;
    return LoadServerConfig(
        yaml_path, AIServerConfigOverrides{}, out_config, report, error);
}
