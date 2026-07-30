#include "model/model_manifest.h"

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>
#include <openssl/evp.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace {

using google::protobuf::Struct;
using google::protobuf::Value;

const Value* FindField(const Struct& object, const std::string& name) {
    const auto& fields = object.fields();
    auto it = fields.find(name);
    return it == fields.end() ? nullptr : &it->second;
}

bool ReadString(const Struct& object, const std::string& name,
                std::string& value, std::string& error) {
    const Value* field = FindField(object, name);
    if (!field || field->kind_case() != Value::kStringValue) {
        error = "manifest field '" + name + "' must be a string";
        return false;
    }
    value = field->string_value();
    return true;
}

bool ReadInteger(const Struct& object, const std::string& name,
                 int64_t& value, std::string& error) {
    const Value* field = FindField(object, name);
    if (!field || field->kind_case() != Value::kNumberValue) {
        error = "manifest field '" + name + "' must be a number";
        return false;
    }
    double raw = field->number_value();
    value = static_cast<int64_t>(raw);
    if (static_cast<double>(value) != raw) {
        error = "manifest field '" + name + "' must be an integer";
        return false;
    }
    return true;
}

bool ReadBoolean(const Struct& object, const std::string& name,
                 bool& value, std::string& error) {
    const Value* field = FindField(object, name);
    if (!field || field->kind_case() != Value::kBoolValue) {
        error = "manifest field '" + name + "' must be a boolean";
        return false;
    }
    value = field->bool_value();
    return true;
}

bool ReadShape(const Struct& object, const std::string& name,
               std::vector<int64_t>& shape, std::string& error) {
    const Value* field = FindField(object, name);
    if (!field || field->kind_case() != Value::kListValue) {
        error = "manifest field '" + name + "' must be an array";
        return false;
    }
    shape.clear();
    for (const auto& item : field->list_value().values()) {
        if (item.kind_case() != Value::kNumberValue) {
            error = "manifest shape '" + name + "' must contain integers";
            return false;
        }
        int64_t dim = static_cast<int64_t>(item.number_value());
        if (static_cast<double>(dim) != item.number_value()) {
            error = "manifest shape '" + name + "' must contain integers";
            return false;
        }
        shape.push_back(dim);
    }
    return true;
}

bool ShapeMatches(const std::vector<int64_t>& actual,
                  const std::vector<int64_t>& expected) {
    return actual == expected;
}

}  // namespace

bool ComputeFileSha256(const std::string& path,
                       std::string& checksum,
                       std::string& error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        error = "cannot open model file: " + path;
        return false;
    }

    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context) {
        error = "cannot allocate SHA-256 context";
        return false;
    }

    bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1;
    std::array<char, 64 * 1024> buffer{};
    while (ok && stream.good()) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        std::streamsize count = stream.gcount();
        if (count > 0) {
            ok = EVP_DigestUpdate(context, buffer.data(),
                                  static_cast<std::size_t>(count)) == 1;
        }
    }
    if (stream.bad()) {
        ok = false;
    }

    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    if (ok) {
        ok = EVP_DigestFinal_ex(context, digest.data(), &digest_size) == 1;
    }
    EVP_MD_CTX_free(context);

    if (!ok) {
        error = "failed to calculate SHA-256: " + path;
        return false;
    }

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_size; ++i) {
        output << std::setw(2) << static_cast<unsigned int>(digest[i]);
    }
    checksum = output.str();
    return true;
}

bool LoadModelManifest(const ModelConfig& config,
                       const std::string& run_id,
                       ModelManifest& manifest,
                       std::string& error) {
    namespace fs = std::filesystem;

    fs::path run_dir = fs::path(config.p2p_dir) / run_id;
    fs::path manifest_path = run_dir / config.manifest_name;
    return LoadModelManifestFile(
        config, manifest_path.string(), run_id, manifest, error);
}

bool LoadModelManifestFile(const ModelConfig& config,
                           const std::string& manifest_file_path,
                           const std::string& expected_run_id,
                           ModelManifest& manifest,
                           std::string& error) {
    namespace fs = std::filesystem;

    fs::path manifest_path = manifest_file_path;
    fs::path run_dir = manifest_path.parent_path();
    std::ifstream stream(manifest_path);
    if (!stream.is_open()) {
        error = "manifest not found: " + manifest_path.string();
        return false;
    }

    std::ostringstream content;
    content << stream.rdbuf();
    Struct document;
    auto status = google::protobuf::util::JsonStringToMessage(content.str(), &document);
    if (!status.ok()) {
        error = "invalid manifest JSON: " + status.ToString();
        return false;
    }

    int64_t schema_version = 0;
    int64_t model_version = -1;
    int64_t size_bytes = 0;
    int64_t seed = 0;
    int64_t published_ts_ms = 0;
    if (!ReadInteger(document, "schema_version", schema_version, error) ||
        !ReadString(document, "contract_version",
                    manifest.contract_version, error) ||
        !ReadString(document, "run_id", manifest.run_id, error) ||
        !ReadInteger(document, "model_version", model_version, error) ||
        !ReadString(document, "artifact_uri", manifest.artifact_uri, error) ||
        !ReadString(document, "model_file", manifest.model_file, error) ||
        !ReadInteger(document, "size_bytes", size_bytes, error) ||
        !ReadString(document, "sha256", manifest.sha256, error) ||
        !ReadShape(document, "input_shape", manifest.input_shape, error) ||
        !ReadShape(document, "action_shape", manifest.action_shape, error) ||
        !ReadShape(document, "value_shape", manifest.value_shape, error) ||
        !ReadInteger(document, "seed", seed, error) ||
        !ReadInteger(document, "published_ts_ms", published_ts_ms, error) ||
        !ReadBoolean(document, "ready", manifest.ready, error)) {
        return false;
    }

    manifest.schema_version = static_cast<int>(schema_version);
    manifest.model_version = static_cast<int>(model_version);
    manifest.size_bytes = size_bytes;
    manifest.seed = seed;
    manifest.published_ts_ms = published_ts_ms;
    manifest.manifest_path = manifest_path.string();

    if (manifest.schema_version != 1) {
        error = "unsupported model manifest schema_version";
        return false;
    }
    if (manifest.contract_version != "0.3.0") {
        error = "unsupported model manifest contract_version";
        return false;
    }
    if (!manifest.ready) {
        error = "model manifest is not ready";
        return false;
    }
    if (!expected_run_id.empty() &&
        manifest.run_id != expected_run_id) {
        error = "model manifest run_id does not match AIServer run_id";
        return false;
    }
    if (manifest.model_version != 0) {
        error = "initial model_version must be 0";
        return false;
    }
    if (manifest.model_file.empty() ||
        fs::path(manifest.model_file).filename() != fs::path(manifest.model_file)) {
        error = "model_file must be a file name inside the run directory";
        return false;
    }
    if (!ShapeMatches(manifest.input_shape, {1, config.expected_obs_dim}) ||
        !ShapeMatches(manifest.action_shape, {1, config.expected_action_dim}) ||
        !ShapeMatches(manifest.value_shape, {1, 1})) {
        error = "model manifest shape does not match AIServer contract";
        return false;
    }

    fs::path model_path = run_dir / manifest.model_file;
    manifest.model_path = model_path.string();
    std::error_code size_error;
    const auto actual_size = fs::file_size(model_path, size_error);
    if (size_error ||
        static_cast<int64_t>(actual_size) != manifest.size_bytes) {
        error = "model size does not match manifest";
        return false;
    }
    std::string actual_checksum;
    if (!ComputeFileSha256(manifest.model_path, actual_checksum, error)) {
        return false;
    }
    if (actual_checksum != manifest.sha256) {
        error = "model checksum does not match manifest";
        return false;
    }
    return true;
}
