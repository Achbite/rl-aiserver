#include "model/model_distributor_client.h"

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>
#include <openssl/evp.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr char kCacheDirectory[] = "cache";
constexpr char kLineagesDirectory[] = "lineages";
constexpr char kLineageIdentityFile[] = "lineage.json";
constexpr int64_t kLineageIdentitySchemaVersion = 1;
constexpr std::uintmax_t kMaxLineageIdentityBytes = 64 * 1024;

using google::protobuf::Struct;
using google::protobuf::Value;

bool IsLowercaseSha256(const std::string& value) {
    if (value.size() != 64) return false;
    return std::all_of(value.begin(), value.end(), [](unsigned char item) {
        return (item >= '0' && item <= '9') ||
               (item >= 'a' && item <= 'f');
    });
}

std::string Sha256Bytes(const std::string& payload) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context) return "";
    bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
              EVP_DigestUpdate(
                  context, payload.data(), payload.size()) == 1;
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int size = 0;
    if (ok) ok = EVP_DigestFinal_ex(context, digest.data(), &size) == 1;
    EVP_MD_CTX_free(context);
    if (!ok) return "";
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < size; ++index) {
        output << std::setw(2)
               << static_cast<unsigned int>(digest[index]);
    }
    return output.str();
}

void WriteJsonString(std::ostream& output, std::string_view value) {
    static constexpr char kHex[] = "0123456789abcdef";
    output << '"';
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
}

std::string LineageIdentityJson(const std::string& lineage_id,
                                const std::string& lineage_key) {
    std::ostringstream output;
    output << "{\"schema_version\":"
           << kLineageIdentitySchemaVersion
           << ",\"model_lineage_id\":";
    WriteJsonString(output, lineage_id);
    output << ",\"lineage_key\":";
    WriteJsonString(output, lineage_key);
    output << "}\n";
    return output.str();
}

const Value* FindField(const Struct& object, const std::string& name) {
    const auto iterator = object.fields().find(name);
    return iterator == object.fields().end() ? nullptr : &iterator->second;
}

bool ReadStringField(const Struct& object,
                     const std::string& name,
                     std::string& value,
                     std::string& error) {
    const Value* field = FindField(object, name);
    if (!field || field->kind_case() != Value::kStringValue) {
        error = "lineage identity field '" + name + "' must be a string";
        return false;
    }
    value = field->string_value();
    return true;
}

bool ReadLineageIdentityFile(const std::filesystem::path& path,
                             const std::string& expected_lineage_id,
                             const std::string& expected_lineage_key,
                             std::string& error) {
    namespace fs = std::filesystem;
    std::error_code fs_error;
    const auto status = fs::symlink_status(path, fs_error);
    if (fs_error || fs::is_symlink(status) || !fs::is_regular_file(status)) {
        error = "lineage identity must be a regular non-symlink file";
        return false;
    }
    const std::uintmax_t size = fs::file_size(path, fs_error);
    if (fs_error || size == 0 || size > kMaxLineageIdentityBytes) {
        error = "lineage identity file size is invalid";
        return false;
    }
    std::ifstream input(path, std::ios::binary);
    std::ostringstream payload;
    payload << input.rdbuf();
    if (!input.good() && !input.eof()) {
        error = "cannot read lineage identity file";
        return false;
    }
    Struct document;
    const auto parse_status =
        google::protobuf::util::JsonStringToMessage(
            payload.str(), &document);
    if (!parse_status.ok() || document.fields_size() != 3) {
        error = "lineage identity JSON is invalid or has unknown fields";
        return false;
    }
    const Value* schema = FindField(document, "schema_version");
    if (!schema || schema->kind_case() != Value::kNumberValue ||
        schema->number_value() !=
            static_cast<double>(kLineageIdentitySchemaVersion)) {
        error = "lineage identity schema_version is invalid";
        return false;
    }
    std::string lineage_id;
    std::string lineage_key;
    if (!ReadStringField(
            document, "model_lineage_id", lineage_id, error) ||
        !ReadStringField(document, "lineage_key", lineage_key, error)) {
        return false;
    }
    if (lineage_id.empty() || !IsLowercaseSha256(lineage_key) ||
        Sha256Bytes(lineage_id) != lineage_key ||
        lineage_id != expected_lineage_id ||
        lineage_key != expected_lineage_key) {
        error = "lineage identity does not match the selected namespace";
        return false;
    }
    error.clear();
    return true;
}

bool ShapeEquals(const google::protobuf::RepeatedField<int64_t>& actual,
                 std::initializer_list<int64_t> expected) {
    if (actual.size() != static_cast<int>(expected.size())) return false;
    int index = 0;
    for (const int64_t dimension : expected) {
        if (actual.Get(index++) != dimension) return false;
    }
    return true;
}

bool IsSha256(const std::string& value) {
    if (value.size() != 64) return false;
    for (const char character : value) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

bool ValidModelDistributorAuthority(
    const common::ServiceInstanceIdentity& identity) {
    return identity.component() == "model-distributor" &&
           !identity.instance_id().empty() &&
           identity.lifecycle_epoch() > 0;
}

bool SameAuthority(const common::ServiceInstanceIdentity& lhs,
                   const common::ServiceInstanceIdentity& rhs) {
    return lhs.component() == rhs.component() &&
           lhs.instance_id() == rhs.instance_id() &&
           lhs.lifecycle_epoch() == rhs.lifecycle_epoch();
}

bool IsRetryableAuthorityTransport(const grpc::Status& status) {
    if (status.ok()) return false;
    switch (status.error_code()) {
        case grpc::StatusCode::ABORTED:
        case grpc::StatusCode::CANCELLED:
        case grpc::StatusCode::DEADLINE_EXCEEDED:
        case grpc::StatusCode::INTERNAL:
        case grpc::StatusCode::RESOURCE_EXHAUSTED:
        case grpc::StatusCode::UNKNOWN:
        case grpc::StatusCode::UNAVAILABLE:
            return true;
        default:
            return false;
    }
}

bool ContractMatchesConfig(const common::ContractIdentity& actual,
                           const ContractConfig& expected) {
    return expected.source_digest.algorithm == "sha256" &&
           expected.artifact_digest.algorithm == "sha256" &&
           actual.package_name() == expected.package_name &&
           actual.package_version() == expected.package_version &&
           actual.source_digest().algorithm() ==
               common::DIGEST_ALGORITHM_SHA256 &&
           actual.source_digest().hex() == expected.source_digest.hex &&
           actual.artifact_digest().algorithm() ==
               common::DIGEST_ALGORITHM_SHA256 &&
           actual.artifact_digest().hex() == expected.artifact_digest.hex &&
           actual.platform() == expected.platform &&
           actual.generator_identity() == expected.generator_identity;
}

bool WriteAll(int descriptor, const char* data, std::size_t size) {
    std::size_t written = 0;
    while (written < size) {
        const ssize_t count =
            ::write(descriptor, data + written, size - written);
        if (count <= 0) return false;
        written += static_cast<std::size_t>(count);
    }
    return true;
}

bool FsyncDirectory(const std::filesystem::path& directory,
                    std::string& error) {
    const int descriptor = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
    if (descriptor < 0) {
        error = "cannot open model cache directory for fsync: " +
                directory.string();
        return false;
    }
    bool ok = ::fsync(descriptor) == 0;
    if (::close(descriptor) != 0) ok = false;
    if (!ok) {
        error = "cannot fsync model cache directory: " + directory.string();
        return false;
    }
    return true;
}

bool IsPrivateTemporaryDirectory(const std::filesystem::path& directory) {
    const std::string name = directory.filename().string();
    return name.rfind(".tmp-", 0) == 0;
}

bool IsPrivatePruneDirectory(const std::filesystem::path& directory) {
    const std::string name = directory.filename().string();
    return name.rfind(".prune-", 0) == 0;
}

bool ParseCanonicalStepDirectory(const std::string& name,
                                 ModelStep& step) {
    if (name.size() < 7 ||
        !std::all_of(name.begin(), name.end(), [](unsigned char value) {
            return std::isdigit(value) != 0;
        })) {
        return false;
    }
    try {
        const unsigned long long parsed = std::stoull(name);
        step = static_cast<ModelStep>(parsed);
    } catch (...) {
        return false;
    }
    std::ostringstream canonical;
    canonical << std::setfill('0') << std::setw(7) << step;
    return canonical.str() == name;
}

bool SameWireManifest(const ModelManifest& left,
                      const ModelManifest& right) {
    return left.wire.SerializeAsString() == right.wire.SerializeAsString();
}

bool IsUnsignedDecimal(const std::string& value) {
    return !value.empty() &&
           std::all_of(value.begin(), value.end(), [](unsigned char item) {
               return std::isdigit(item) != 0;
           });
}

bool ParsePrivateCacheDirectory(const std::string& name,
                                const std::string& prefix,
                                ModelStep& step) {
    if (name.rfind(prefix, 0) != 0) return false;
    const std::string remainder = name.substr(prefix.size());
    const std::size_t last_separator = remainder.rfind('-');
    if (last_separator == std::string::npos ||
        last_separator == 0 ||
        last_separator + 1 >= remainder.size()) {
        return false;
    }
    const std::size_t process_separator =
        remainder.rfind('-', last_separator - 1);
    if (process_separator == std::string::npos ||
        process_separator == 0 ||
        process_separator + 1 == last_separator) {
        return false;
    }
    return ParseCanonicalStepDirectory(
               remainder.substr(0, process_separator), step) &&
           IsUnsignedDecimal(remainder.substr(
               process_separator + 1,
               last_separator - process_separator - 1)) &&
           IsUnsignedDecimal(remainder.substr(last_separator + 1));
}

bool ParsePrivateLineageIdentityFile(const std::string& name) {
    constexpr std::string_view kPrefix = ".lineage-";
    constexpr std::string_view kSuffix = ".tmp";
    if (name.size() <= kPrefix.size() + kSuffix.size() ||
        name.compare(0, kPrefix.size(), kPrefix) != 0 ||
        name.compare(name.size() - kSuffix.size(),
                     kSuffix.size(), kSuffix) != 0) {
        return false;
    }
    const std::string identity = name.substr(
        kPrefix.size(),
        name.size() - kPrefix.size() - kSuffix.size());
    const std::size_t separator = identity.find('-');
    return separator != std::string::npos && separator > 0 &&
           separator + 1 < identity.size() &&
           identity.find('-', separator + 1) == std::string::npos &&
           IsUnsignedDecimal(identity.substr(0, separator)) &&
           IsUnsignedDecimal(identity.substr(separator + 1));
}

bool WriteLineageIdentityFile(const std::filesystem::path& directory,
                              const std::string& lineage_id,
                              const std::string& lineage_key,
                              std::string& error) {
    namespace fs = std::filesystem;
    static std::atomic<uint64_t> next_identity_id{1};
    const fs::path final_path = directory / kLineageIdentityFile;
    std::error_code fs_error;
    if (fs::exists(final_path, fs_error)) {
        return ReadLineageIdentityFile(
            final_path, lineage_id, lineage_key, error);
    }
    if (fs_error) {
        error = "cannot inspect lineage identity file: " +
                fs_error.message();
        return false;
    }
    const fs::path temporary_path =
        directory /
        (".lineage-" + std::to_string(::getpid()) + "-" +
         std::to_string(next_identity_id.fetch_add(1)) + ".tmp");
    const std::string payload =
        LineageIdentityJson(lineage_id, lineage_key);
    const int descriptor = ::open(
        temporary_path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
    if (descriptor < 0) {
        error = "cannot create private lineage identity file";
        return false;
    }
    bool ok = WriteAll(descriptor, payload.data(), payload.size()) &&
              ::fsync(descriptor) == 0;
    if (::close(descriptor) != 0) ok = false;
    if (!ok) {
        fs::remove(temporary_path, fs_error);
        error = "cannot persist private lineage identity file";
        return false;
    }
    fs::rename(temporary_path, final_path, fs_error);
    if (fs_error) {
        std::error_code remove_error;
        fs::remove(temporary_path, remove_error);
        error = "cannot atomically publish lineage identity file: " +
                fs_error.message();
        return false;
    }
    if (!FsyncDirectory(directory, error) ||
        !ReadLineageIdentityFile(
            final_path, lineage_id, lineage_key, error)) {
        return false;
    }
    error.clear();
    return true;
}

bool EnsureCacheRoot(const AIServerConfig& config,
                     std::filesystem::path& cache_root,
                     std::string& error) {
    namespace fs = std::filesystem;
    cache_root =
        fs::path(config.model.local_train_dir) / kCacheDirectory;
    std::error_code fs_error;
    fs::create_directories(cache_root, fs_error);
    if (fs_error) {
        error = "cannot create model cache directory: " +
                fs_error.message();
        return false;
    }
    const auto status = fs::symlink_status(cache_root, fs_error);
    if (fs_error || fs::is_symlink(status) ||
        !fs::is_directory(status)) {
        error = "model cache root must be a real directory";
        return false;
    }
    return true;
}

bool InspectLegacyCacheRoot(const std::filesystem::path& cache_root,
                            std::size_t& ignored_entries,
                            std::string& error) {
    namespace fs = std::filesystem;
    ignored_entries = 0;
    std::error_code fs_error;
    for (fs::directory_iterator iterator(cache_root, fs_error), end;
         !fs_error && iterator != end; iterator.increment(fs_error)) {
        const fs::path path = iterator->path();
        const std::string name = path.filename().string();
        const auto status = fs::symlink_status(path, fs_error);
        if (fs_error) break;
        if (name == kLineagesDirectory) {
            if (fs::is_symlink(status) || !fs::is_directory(status)) {
                error = "model cache lineages root must be a real directory";
                return false;
            }
            continue;
        }
        ModelStep ignored_step = 0;
        const bool recognized =
            ParseCanonicalStepDirectory(name, ignored_step) ||
            ParsePrivateCacheDirectory(name, ".tmp-", ignored_step) ||
            ParsePrivateCacheDirectory(name, ".prune-", ignored_step);
        if (!recognized || fs::is_symlink(status) ||
            !fs::is_directory(status)) {
            error = "model cache contains an unrecognized legacy entry: " +
                    name;
            return false;
        }
        ++ignored_entries;
    }
    if (fs_error) {
        error = "cannot inspect legacy model cache: " +
                fs_error.message();
        return false;
    }
    error.clear();
    return true;
}

bool EnsureActiveLineageNamespace(
    const AIServerConfig& config,
    const std::string& lineage_id,
    const std::string& lineage_key,
    std::filesystem::path& active_root,
    std::size_t& ignored_legacy_entries,
    std::string& error) {
    namespace fs = std::filesystem;
    if (lineage_id.empty() || !IsLowercaseSha256(lineage_key) ||
        Sha256Bytes(lineage_id) != lineage_key) {
        error = "selected model lineage identity is invalid";
        return false;
    }
    fs::path cache_root;
    if (!EnsureCacheRoot(config, cache_root, error) ||
        !InspectLegacyCacheRoot(
            cache_root, ignored_legacy_entries, error)) {
        return false;
    }
    const fs::path lineages_root = cache_root / kLineagesDirectory;
    std::error_code fs_error;
    const bool created_lineages =
        fs::create_directory(lineages_root, fs_error);
    if (fs_error) {
        error = "cannot create model cache lineages directory: " +
                fs_error.message();
        return false;
    }
    const auto lineages_status =
        fs::symlink_status(lineages_root, fs_error);
    if (fs_error || fs::is_symlink(lineages_status) ||
        !fs::is_directory(lineages_status)) {
        error = "model cache lineages root must be a real directory";
        return false;
    }
    active_root = lineages_root / lineage_key;
    const bool created_active = fs::create_directory(active_root, fs_error);
    if (fs_error) {
        error = "cannot create active model lineage cache: " +
                fs_error.message();
        return false;
    }
    const auto active_status = fs::symlink_status(active_root, fs_error);
    if (fs_error || fs::is_symlink(active_status) ||
        !fs::is_directory(active_status)) {
        error = "active model lineage cache must be a real directory";
        return false;
    }
    if (created_active) {
        if (!WriteLineageIdentityFile(
                active_root, lineage_id, lineage_key, error) ||
            !FsyncDirectory(lineages_root, error)) {
            return false;
        }
    } else if (!ReadLineageIdentityFile(
                   active_root / kLineageIdentityFile,
                   lineage_id, lineage_key, error)) {
        return false;
    }
    if (created_lineages && !FsyncDirectory(cache_root, error)) {
        return false;
    }
    error.clear();
    return true;
}

bool OpenActiveLineageNamespace(
    const AIServerConfig& config,
    const std::string& lineage_id,
    const std::string& lineage_key,
    std::filesystem::path& active_root,
    std::string& error) {
    namespace fs = std::filesystem;
    if (lineage_id.empty() || !IsLowercaseSha256(lineage_key) ||
        Sha256Bytes(lineage_id) != lineage_key) {
        error = "selected model lineage identity is invalid";
        return false;
    }
    fs::path cache_root;
    if (!EnsureCacheRoot(config, cache_root, error)) return false;
    const fs::path lineages_root = cache_root / kLineagesDirectory;
    std::error_code fs_error;
    const auto lineages_status =
        fs::symlink_status(lineages_root, fs_error);
    if (fs_error || fs::is_symlink(lineages_status) ||
        !fs::is_directory(lineages_status)) {
        error = "model cache lineages root is unavailable";
        return false;
    }
    active_root = lineages_root / lineage_key;
    const auto active_status = fs::symlink_status(active_root, fs_error);
    if (fs_error || fs::is_symlink(active_status) ||
        !fs::is_directory(active_status)) {
        error = "active model lineage cache is unavailable";
        return false;
    }
    return ReadLineageIdentityFile(
        active_root / kLineageIdentityFile,
        lineage_id, lineage_key, error);
}

bool RemoveValidatedPrivateDirectory(
    const std::filesystem::path& cache_root,
    const std::filesystem::path& directory,
    const std::string& prefix,
    std::string& error) {
    namespace fs = std::filesystem;
    ModelStep ignored_step = 0;
    std::error_code fs_error;
    const auto status = fs::symlink_status(directory, fs_error);
    if (fs_error || fs::is_symlink(status) || !fs::is_directory(status) ||
        directory.parent_path() != cache_root ||
        !ParsePrivateCacheDirectory(
            directory.filename().string(), prefix, ignored_step)) {
        error = "refusing to remove an invalid private model directory: " +
                directory.string();
        return false;
    }
    fs::remove_all(directory, fs_error);
    if (fs_error) {
        error = "cannot remove private model directory: " +
                fs_error.message();
        return false;
    }
    return true;
}

std::filesystem::path AllocatePrivateCacheDirectory(
    const std::filesystem::path& cache_root,
    const std::string& prefix,
    ModelStep model_step,
    std::string& error) {
    namespace fs = std::filesystem;
    static std::atomic<uint64_t> next_private_id{1};
    std::error_code fs_error;
    for (int attempt = 0; attempt < 100; ++attempt) {
        const fs::path candidate =
            cache_root /
            (prefix + ModelDistributorClient::CacheStepDirectoryName(
                          model_step) +
             "-" + std::to_string(::getpid()) + "-" +
             std::to_string(next_private_id.fetch_add(1)));
        if (fs::create_directory(candidate, fs_error)) return candidate;
        if (fs_error && fs_error != std::errc::file_exists) {
            error = "cannot create private model cache directory: " +
                    fs_error.message();
            return {};
        }
        fs_error.clear();
    }
    error = "cannot allocate private model cache directory";
    return {};
}

}  // namespace

ModelDistributorClient::ModelDistributorClient(
    const AIServerConfig& config,
    std::string producer_instance_id,
    uint64_t producer_lifecycle_epoch)
    : config_(config) {
    requester_identity_.set_component("rl-aiserver");
    requester_identity_.set_instance_id(std::move(producer_instance_id));
    requester_identity_.set_lifecycle_epoch(producer_lifecycle_epoch);
    const std::string address =
        config_.model_distribution.host + ":" +
        std::to_string(config_.model_distribution.port);
    channel_ = grpc::CreateChannel(
        address, grpc::InsecureChannelCredentials());
    stub_ = training::ModelDistributorService::NewStub(channel_);
}

bool ModelDistributorClient::ValidateManifest(
    const training::ModelArtifactManifest& source,
    std::optional<ModelStep> expected_step,
    std::string& error) const {
    if (!ValidateModelManifest(config_, source, expected_step, error)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(lineage_mutex_);
    if (!pinned_model_lineage_id_.has_value() ||
        source.identity().model_lineage_id() !=
            *pinned_model_lineage_id_) {
        error = "model manifest does not match the discovered training lineage";
        return false;
    }
    return true;
}

bool ModelDistributorClient::PinModelLineage(
    const std::string& lineage_id,
    std::string& error) {
    if (lineage_id.empty()) {
        error = "Model Distributor latest model lineage is missing";
        return false;
    }
    const std::string lineage_key = Sha256Bytes(lineage_id);
    if (!IsLowercaseSha256(lineage_key)) {
        error = "cannot derive model lineage cache key";
        return false;
    }
    std::lock_guard<std::mutex> lock(lineage_mutex_);
    if (pinned_model_lineage_id_.has_value() &&
        *pinned_model_lineage_id_ != lineage_id) {
        error = "model lineage changed within the isolated training workspace";
        return false;
    }
    if (pinned_model_lineage_key_.has_value() &&
        *pinned_model_lineage_key_ != lineage_key) {
        error = "model lineage cache key changed within the process lifecycle";
        return false;
    }
    pinned_model_lineage_id_ = lineage_id;
    pinned_model_lineage_key_ = lineage_key;
    error.clear();
    return true;
}

bool ModelDistributorClient::GetPinnedModelLineage(
    std::string& lineage_id,
    std::string& lineage_key,
    std::string& error) const {
    std::lock_guard<std::mutex> lock(lineage_mutex_);
    if (!pinned_model_lineage_id_.has_value() ||
        !pinned_model_lineage_key_.has_value()) {
        error = "model lineage must be discovered before cache access";
        return false;
    }
    lineage_id = *pinned_model_lineage_id_;
    lineage_key = *pinned_model_lineage_key_;
    if (lineage_id.empty() || !IsLowercaseSha256(lineage_key) ||
        Sha256Bytes(lineage_id) != lineage_key) {
        error = "pinned model lineage identity is invalid";
        return false;
    }
    error.clear();
    return true;
}

bool ModelDistributorClient::DownloadToTemporary(
    const training::ModelArtifactManifest& source,
    const std::string& aiserver_id,
    std::string& local_path,
    std::string& error) {
    namespace fs = std::filesystem;
    if (source.model_file() != kCachedModelFile) {
        error = "distributed model_file must be SaveModel.onnx";
        return false;
    }
    std::string lineage_id;
    std::string lineage_key;
    fs::path cache_root;
    if (!GetPinnedModelLineage(lineage_id, lineage_key, error) ||
        source.identity().model_lineage_id() != lineage_id ||
        !OpenActiveLineageNamespace(
            config_, lineage_id, lineage_key, cache_root, error)) {
        if (error.empty()) {
            error = "distributed model does not match the active lineage cache";
        }
        return false;
    }
    std::error_code fs_error;

    const fs::path temporary_dir = AllocatePrivateCacheDirectory(
        cache_root, ".tmp-",
        source.identity().model_step(), error);
    if (temporary_dir.empty()) {
        return false;
    }

    const fs::path temporary_model = temporary_dir / kCachedModelFile;
    const int descriptor = ::open(
        temporary_model.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
    if (descriptor < 0) {
        fs::remove_all(temporary_dir, fs_error);
        error = "cannot open model download temporary file";
        return false;
    }

    training::DownloadModelReq request;
    *request.mutable_requested_model() = source.identity();
    request.mutable_requester()->CopyFrom(requester_identity_);
    grpc::ClientContext context;
    context.set_deadline(
        std::chrono::system_clock::now() +
        std::chrono::milliseconds(config_.model_distribution.rpc_timeout_ms));
    std::unique_ptr<grpc::ClientReader<training::ModelChunk>> reader =
        stub_->DownloadModel(&context, request);
    int64_t expected_offset = 0;
    training::ModelChunk chunk;
    bool write_ok = true;
    while (reader->Read(&chunk)) {
        if (chunk.model().SerializeAsString() !=
                source.identity().SerializeAsString() ||
            chunk.offset() != expected_offset ||
            !WriteAll(descriptor, chunk.data().data(), chunk.data().size())) {
            write_ok = false;
            break;
        }
        expected_offset += static_cast<int64_t>(chunk.data().size());
    }
    const grpc::Status status = reader->Finish();
    if (::fsync(descriptor) != 0) write_ok = false;
    ::close(descriptor);

    if (!status.ok() || !write_ok ||
        expected_offset != source.size_bytes()) {
        fs::remove_all(temporary_dir, fs_error);
        error = status.ok()
                    ? "model stream offset or size mismatch"
                    : "model download failed: " + status.error_message();
        return false;
    }
    std::string checksum;
    if (!ComputeFileSha256(temporary_model.string(), checksum, error)) {
        fs::remove_all(temporary_dir, fs_error);
        return false;
    }
    if (checksum != source.identity().artifact_digest().hex()) {
        fs::remove_all(temporary_dir, fs_error);
        error = "downloaded model checksum mismatch";
        return false;
    }

    const fs::path manifest_path =
        temporary_dir / kModelManifestFile;
    if (!WriteModelManifestFile(source, manifest_path.string(), error) ||
        !FsyncDirectory(temporary_dir, error)) {
        fs::remove_all(temporary_dir, fs_error);
        return false;
    }

    ModelManifest validated;
    if (!LoadModelManifestFile(
            config_, manifest_path.string(), validated, error) ||
        validated.wire.SerializeAsString() != source.SerializeAsString()) {
        fs::remove_all(temporary_dir, fs_error);
        if (error.empty()) {
            error = "persisted model manifest identity mismatch";
        }
        return false;
    }
    local_path = temporary_model.string();
    return true;
}

bool ModelDistributorClient::Fetch(const std::string& aiserver_id,
                                   ModelStep model_step,
                                   bool latest,
                                   ModelManifest& manifest,
    std::string& error) {
    AvailableRange discovered;
    bool has_pinned_lineage = false;
    {
        std::lock_guard<std::mutex> lock(lineage_mutex_);
        has_pinned_lineage = pinned_model_lineage_id_.has_value();
    }
    if (latest || !has_pinned_lineage) {
        if (!GetAvailableRange(aiserver_id, discovered, error)) return false;
        if (latest) model_step = discovered.latest_model_step;
    }
    std::string lineage_id;
    {
        std::lock_guard<std::mutex> lock(lineage_mutex_);
        if (!pinned_model_lineage_id_.has_value()) {
            error = "model lineage discovery did not produce an identity";
            return false;
        }
        lineage_id = *pinned_model_lineage_id_;
    }
    training::GetModelManifestReq request;
    request.mutable_requested_model()->set_model_lineage_id(
        lineage_id);
    request.mutable_requested_model()->set_model_step(model_step);
    request.mutable_requester()->CopyFrom(requester_identity_);
    request.set_latest_in_lineage(false);
    training::GetModelManifestRsp response;
    grpc::ClientContext context;
    context.set_deadline(
        std::chrono::system_clock::now() +
        std::chrono::milliseconds(config_.model_distribution.rpc_timeout_ms));
    const grpc::Status status =
        stub_->GetModelManifest(&context, request, &response);
    if (!status.ok() || response.ret_code() != 0 ||
        !response.has_manifest()) {
        error = status.ok()
                    ? response.message()
                    : "model manifest RPC failed: " + status.error_message();
        return false;
    }
    const auto& source = response.manifest();
    if (!ValidateManifest(
            source, std::optional<ModelStep>(model_step), error)) {
        return false;
    }

    std::string local_path;
    if (!DownloadToTemporary(source, aiserver_id, local_path, error)) {
        return false;
    }
    const std::filesystem::path manifest_path =
        std::filesystem::path(local_path).parent_path() /
        kModelManifestFile;
    if (!LoadModelManifestFile(
            config_, manifest_path.string(), manifest, error) ||
        manifest.wire.SerializeAsString() != source.SerializeAsString()) {
        std::error_code remove_error;
        std::filesystem::remove_all(
            std::filesystem::path(local_path).parent_path(), remove_error);
        if (error.empty()) error = "downloaded model identity mismatch";
        return false;
    }
    error.clear();
    return true;
}

bool ModelDistributorClient::FetchLatest(
    const std::string& aiserver_id,
    ModelManifest& manifest,
    std::string& error) {
    return Fetch(aiserver_id, 0, true, manifest, error);
}

bool ModelDistributorClient::FetchStep(
    const std::string& aiserver_id,
    ModelStep model_step,
    ModelManifest& manifest,
    std::string& error) {
    return Fetch(
        aiserver_id, model_step, false, manifest, error);
}

bool ModelDistributorClient::GetLatestIdentity(
    const std::string& aiserver_id,
    ModelStep& model_step,
    std::string& checksum,
    std::string& error) {
    AvailableRange range;
    if (!GetAvailableRange(aiserver_id, range, error)) return false;
    model_step = range.latest_model_step;
    checksum = range.latest_checksum;
    return true;
}

bool ModelDistributorClient::GetAvailableRange(
    const std::string& aiserver_id,
    AvailableRange& range,
    std::string& error) {
    (void)aiserver_id;
    range = AvailableRange{};
    training::ModelDistributorStatusReq request;
    training::ModelDistributorStatusRsp response;
    grpc::ClientContext context;
    context.set_deadline(
        std::chrono::system_clock::now() +
        std::chrono::milliseconds(config_.model_distribution.rpc_timeout_ms));
    const grpc::Status status =
        stub_->GetModelDistributorStatus(&context, request, &response);
    if (!status.ok()) {
        error = "model distributor status RPC failed: " +
                status.error_message();
        return false;
    }
    if (!response.ready()) {
        error = "Model Distributor status is not ready";
        return false;
    }
    if (!ValidModelDistributorAuthority(response.distributor())) {
        error = "Model Distributor status authority is invalid";
        return false;
    }
    if (!ContractMatchesConfig(response.contract(), config_.contract)) {
        error = "Model Distributor status contract does not match the configured rl-contract";
        return false;
    }
    if (!response.has_latest_model() ||
        response.latest_model().model_lineage_id().empty() ||
        !IsSha256(response.latest_model().artifact_digest().hex()) ||
        !IsSha256(response.latest_model().manifest_digest().hex())) {
        error = "Model Distributor latest model identity is missing or invalid";
        return false;
    }
    const auto& latest = response.latest_model();
    if (!PinModelLineage(latest.model_lineage_id(), error)) return false;

    if (!latest.has_model_step() ||
        !response.has_available_floor_model_step() ||
        !response.has_latest_available_model_step()) {
        error = "Model Distributor available range is missing";
        return false;
    }

    const uint64_t floor_step = response.available_floor_model_step();
    const uint64_t latest_step = response.latest_available_model_step();
    if (floor_step > latest_step || latest_step != latest.model_step()) {
        error = "Model Distributor available range is invalid";
        return false;
    }
    range.floor_model_step = floor_step;
    range.latest_model_step = latest_step;
    range.model_lineage_id = latest.model_lineage_id();
    range.latest_checksum = latest.artifact_digest().hex();
    range.latest_manifest_digest = latest.manifest_digest().hex();
    error.clear();
    return true;
}

bool ModelDistributorClient::Ack(const ModelManifest& manifest,
                                 const std::string& aiserver_id,
                                 training::ModelLoadStatus load_status,
                                 const std::string& message,
                                 std::string& error) {
    return AckIdempotently(
               manifest, aiserver_id, load_status, message, error) ==
           AckDisposition::Applied;
}

bool ModelDistributorClient::ProbeAckAuthority(
    common::ServiceInstanceIdentity& authority,
    std::string& error) {
    return ProbeAckAuthorityDisposition(authority, error) ==
           AuthorityProbeDisposition::Ready;
}

ModelDistributorClient::AuthorityProbeDisposition
ModelDistributorClient::ProbeAckAuthorityDisposition(
    common::ServiceInstanceIdentity& authority,
    std::string& error) {
    training::ModelDistributorStatusReq request;
    training::ModelDistributorStatusRsp response;
    grpc::ClientContext context;
    context.set_deadline(
        std::chrono::system_clock::now() +
        std::chrono::milliseconds(config_.model_distribution.rpc_timeout_ms));
    const grpc::Status status =
        stub_->GetModelDistributorStatus(&context, request, &response);
    if (!status.ok()) {
        error = "model ACK authority probe failed: " +
                status.error_message();
        authority.Clear();
        return IsRetryableAuthorityTransport(status)
                   ? AuthorityProbeDisposition::Retryable
                   : AuthorityProbeDisposition::Rejected;
    }
    if (!response.ready()) {
        error = "model ACK authority is not ready";
        authority.Clear();
        return AuthorityProbeDisposition::Retryable;
    }
    if (!ValidModelDistributorAuthority(response.distributor())) {
        error = "model ACK authority identity is invalid";
        authority.Clear();
        return AuthorityProbeDisposition::Rejected;
    }
    if (!ContractMatchesConfig(response.contract(), config_.contract)) {
        error = "model ACK authority contract does not match the configured "
                "rl-contract";
        authority.Clear();
        return AuthorityProbeDisposition::Rejected;
    }
    authority.CopyFrom(response.distributor());
    error.clear();
    return AuthorityProbeDisposition::Ready;
}

ModelDistributorClient::AckDisposition
ModelDistributorClient::AckIdempotently(
    const ModelManifest& manifest,
    const std::string& aiserver_id,
    training::ModelLoadStatus load_status,
    const std::string& message,
    std::string& error,
    common::ServiceInstanceIdentity* pinned_authority) {
    common::ServiceInstanceIdentity local_authority;
    auto* authority = pinned_authority ? pinned_authority : &local_authority;
    if (!ValidModelDistributorAuthority(*authority)) {
        if (!ProbeAckAuthority(*authority, error)) {
            return AckDisposition::NotApplied;
        }
    }

    training::AckModelReq request;
    request.mutable_aiserver()->CopyFrom(requester_identity_);
    *request.mutable_model() = manifest.wire.identity();
    request.set_load_instance_id(
        requester_identity_.instance_id() + "-load-step-" +
        std::to_string(manifest.model_step));
    request.set_load_status(load_status);
    request.set_message(message);
    constexpr int kMaxAttempts = 3;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        training::AckModelRsp response;
        grpc::ClientContext context;
        context.set_deadline(
            std::chrono::system_clock::now() +
            std::chrono::milliseconds(
                config_.model_distribution.rpc_timeout_ms));
        const grpc::Status rpc_status =
            stub_->AckModel(&context, request, &response);
        if (rpc_status.ok()) {
            if (!ValidModelDistributorAuthority(response.distributor())) {
                error = "model ACK response authority is invalid";
                return AckDisposition::Uncertain;
            }
            const bool applied =
                response.result() == training::MODEL_ACK_RESULT_APPLIED ||
                response.result() ==
                    training::MODEL_ACK_RESULT_ALREADY_APPLIED;
            const bool rejected =
                response.result() == training::MODEL_ACK_RESULT_NOT_FOUND ||
                response.result() == training::MODEL_ACK_RESULT_REJECTED;
            if ((response.ret_code() == 0) != applied) {
                error = "model ACK response code and result disagree";
                return AckDisposition::Uncertain;
            }
            if (applied) {
                error.clear();
                return AckDisposition::Applied;
            }
            error = response.message().empty()
                        ? "model ACK was explicitly rejected"
                        : response.message();
            if (rejected &&
                SameAuthority(response.distributor(), *authority)) {
                return AckDisposition::Rejected;
            }
            return AckDisposition::Uncertain;
        }
        error = "model ACK RPC outcome is uncertain: " +
                rpc_status.error_message();
        if (attempt + 1 < kMaxAttempts) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(50 * (attempt + 1)));
        }
    }
    return AckDisposition::Uncertain;
}

std::string ModelDistributorClient::CacheStepDirectoryName(
    ModelStep model_step) {
    std::ostringstream output;
    output << std::setfill('0') << std::setw(7) << model_step;
    return output.str();
}

bool ModelDistributorClient::LoadCachedStep(
    ModelStep model_step,
    ModelManifest& manifest,
    std::string& error) const {
    namespace fs = std::filesystem;
    std::string lineage_id;
    std::string lineage_key;
    fs::path cache_root;
    if (!GetPinnedModelLineage(lineage_id, lineage_key, error) ||
        !OpenActiveLineageNamespace(
            config_, lineage_id, lineage_key, cache_root, error)) {
        return false;
    }
    const fs::path directory =
        cache_root / CacheStepDirectoryName(model_step);
    std::error_code fs_error;
    const auto directory_status = fs::symlink_status(directory, fs_error);
    if (fs_error || fs::is_symlink(directory_status) ||
        !fs::is_directory(directory_status)) {
        error = "cached model step directory is invalid";
        return false;
    }
    std::set<std::string> entries;
    for (fs::directory_iterator iterator(directory, fs_error), end;
         !fs_error && iterator != end; iterator.increment(fs_error)) {
        const auto entry_status =
            fs::symlink_status(iterator->path(), fs_error);
        if (fs_error || fs::is_symlink(entry_status) ||
            !fs::is_regular_file(entry_status)) {
            error = "cached model step contains an invalid entry";
            return false;
        }
        entries.insert(iterator->path().filename().string());
    }
    const std::set<std::string> expected_entries{
        kCachedModelFile, kModelManifestFile};
    if (fs_error || entries != expected_entries) {
        error = "cached model step is not a complete two-file artifact";
        return false;
    }
    if (!LoadModelManifestFile(
            config_,
            (directory / kModelManifestFile).string(),
            manifest, error)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(lineage_mutex_);
        if (pinned_model_lineage_id_.has_value() &&
            manifest.model_lineage_id != *pinned_model_lineage_id_) {
            error = "cached model does not match the discovered training lineage";
            return false;
        }
    }
    if (manifest.model_step != model_step ||
        manifest.model_file != kCachedModelFile ||
        fs::path(manifest.model_path).filename() != kCachedModelFile) {
        error = "cached model directory identity mismatch";
        return false;
    }
    error.clear();
    return true;
}

bool ModelDistributorClient::ListCachedModels(
    std::vector<ModelManifest>& models,
    std::string& error) const {
    namespace fs = std::filesystem;
    models.clear();
    std::string lineage_id;
    std::string lineage_key;
    fs::path cache_root;
    if (!GetPinnedModelLineage(lineage_id, lineage_key, error) ||
        !OpenActiveLineageNamespace(
            config_, lineage_id, lineage_key, cache_root, error)) {
        return false;
    }
    std::error_code fs_error;
    for (fs::directory_iterator iterator(cache_root, fs_error), end;
         !fs_error && iterator != end; iterator.increment(fs_error)) {
        const fs::path path = iterator->path();
        const auto status = fs::symlink_status(path, fs_error);
        if (fs_error) break;
        ModelStep directory_step = 0;
        const std::string name = path.filename().string();
        if (name == kLineageIdentityFile) {
            if (fs::is_symlink(status) || !fs::is_regular_file(status)) {
                error = "active model lineage identity entry is invalid";
                models.clear();
                return false;
            }
            continue;
        }
        if (ParsePrivateCacheDirectory(name, ".tmp-", directory_step)) {
            if (fs::is_symlink(status) || !fs::is_directory(status)) {
                error = "model cache contains an invalid private download: " +
                        name;
                models.clear();
                return false;
            }
            continue;
        }
        if (IsPrivateTemporaryDirectory(path)) {
            error = "model cache contains a malformed private download: " +
                    name;
            models.clear();
            return false;
        }
        if (fs::is_symlink(status) || !fs::is_directory(status) ||
            !ParseCanonicalStepDirectory(
                name, directory_step)) {
            error = "model cache contains an unrecognized entry: " +
                    name;
            models.clear();
            return false;
        }
        ModelManifest candidate;
        std::string validation_error;
        if (!LoadCachedStep(
                directory_step, candidate, validation_error)) {
            error = "cached model step is invalid: " +
                    path.filename().string() + ": " + validation_error;
            models.clear();
            return false;
        }
        models.push_back(std::move(candidate));
    }
    if (fs_error) {
        error = "cannot scan model cache: " + fs_error.message();
        models.clear();
        return false;
    }
    std::sort(models.begin(), models.end(), [](const auto& left,
                                                const auto& right) {
        return left.model_step < right.model_step;
    });
    error.clear();
    return true;
}

bool ModelDistributorClient::RecoverCache(
    std::vector<ModelManifest>& models,
    CacheRecoveryFacts& facts,
    std::string& error) {
    namespace fs = std::filesystem;
    models.clear();
    facts = CacheRecoveryFacts{};
    std::string lineage_id;
    std::string lineage_key;
    fs::path cache_root;
    if (!GetPinnedModelLineage(lineage_id, lineage_key, error) ||
        !EnsureActiveLineageNamespace(
            config_, lineage_id, lineage_key, cache_root,
            facts.ignored_legacy_entries, error)) {
        return false;
    }
    facts.model_lineage_key = lineage_key;

    std::error_code fs_error;
    std::vector<fs::path> entries;
    for (fs::directory_iterator iterator(cache_root, fs_error), end;
         !fs_error && iterator != end; iterator.increment(fs_error)) {
        entries.push_back(iterator->path());
    }
    if (fs_error) {
        error = "cannot scan model cache: " + fs_error.message();
        return false;
    }
    std::sort(entries.begin(), entries.end());
    bool changed = false;
    for (const auto& path : entries) {
        const std::string name = path.filename().string();
        const auto status = fs::symlink_status(path, fs_error);
        if (fs_error) {
            error = "cannot inspect active model cache entry: " + name;
            return false;
        }
        if (name == kLineageIdentityFile) {
            if (fs::is_symlink(status) || !fs::is_regular_file(status) ||
                !ReadLineageIdentityFile(
                    path, lineage_id, lineage_key, error)) {
                if (error.empty()) {
                    error = "active model lineage identity entry is invalid";
                }
                return false;
            }
            continue;
        }
        if (ParsePrivateLineageIdentityFile(name)) {
            if (fs::is_symlink(status) || !fs::is_regular_file(status)) {
                error = "active model cache contains an invalid private lineage identity";
                return false;
            }
            fs::remove(path, fs_error);
            if (fs_error) {
                error = "cannot remove private lineage identity residue: " +
                        fs_error.message();
                return false;
            }
            changed = true;
            continue;
        }
        ModelStep private_step = 0;
        if (ParsePrivateCacheDirectory(name, ".tmp-", private_step)) {
            if (!RemoveValidatedPrivateDirectory(
                    cache_root, path, ".tmp-", error)) {
                return false;
            }
            changed = true;
            continue;
        }
        if (ParsePrivateCacheDirectory(name, ".prune-", private_step)) {
            if (!RemoveValidatedPrivateDirectory(
                    cache_root, path, ".prune-", error)) {
                return false;
            }
            changed = true;
            continue;
        }

        ModelStep directory_step = 0;
        if (fs::is_symlink(status) || !fs::is_directory(status) ||
            !ParseCanonicalStepDirectory(name, directory_step)) {
            error = "model cache contains an unrecognized entry: " + name;
            return false;
        }
        ModelManifest candidate;
        std::string validation_error;
        if (!LoadCachedStep(
                directory_step, candidate, validation_error)) {
            error = "active cached model step is invalid and was preserved: " +
                    name + ": " + validation_error;
            return false;
        }
    }
    if (changed && !FsyncDirectory(cache_root, error)) return false;
    if (!ListCachedModels(models, error)) return false;
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cached_steps_.clear();
        for (const auto& model : models) {
            cached_steps_.insert(model.model_step);
        }
    }
    // Recovery runs before an active/staged inference session exists, so the
    // service has no canonical cache entries to protect yet.
    if (!PruneCache({}, error) || !ListCachedModels(models, error)) return false;
    for (const auto& model : models) {
        if (model.model_lineage_id != lineage_id) {
            error = "active model cache contains a different training lineage";
            return false;
        }
    }
    facts.recovered_steps = models.size();
    error.clear();
    return true;
}

bool ModelDistributorClient::PublishPrepared(
    ModelManifest& manifest,
    std::string& error) {
    namespace fs = std::filesystem;
    if (manifest.model_file != kCachedModelFile) {
        error = "prepared model identity is invalid";
        return false;
    }
    std::string lineage_id;
    std::string lineage_key;
    fs::path cache_root;
    if (!GetPinnedModelLineage(lineage_id, lineage_key, error) ||
        manifest.model_lineage_id != lineage_id ||
        !OpenActiveLineageNamespace(
            config_, lineage_id, lineage_key, cache_root, error)) {
        if (error.empty()) {
            error = "prepared model does not match the active lineage cache";
        }
        return false;
    }
    const fs::path temporary_model = manifest.model_path;
    const fs::path temporary_dir = temporary_model.parent_path();
    const fs::path final_dir =
        cache_root / CacheStepDirectoryName(manifest.model_step);
    std::error_code fs_error;
    if (temporary_model.filename() != kCachedModelFile ||
        !IsPrivateTemporaryDirectory(temporary_dir) ||
        !fs::equivalent(temporary_dir.parent_path(), cache_root, fs_error) ||
        fs_error) {
        error = "prepared model is not in a private cache directory";
        return false;
    }
    ModelManifest validated;
    if (!LoadModelManifestFile(
            config_,
            (temporary_dir / kModelManifestFile).string(),
            validated, error) ||
        !SameWireManifest(validated, manifest) ||
        validated.model_step != manifest.model_step) {
        if (error.empty()) error = "prepared model identity changed";
        return false;
    }

    if (fs::exists(final_dir, fs_error)) {
        if (fs_error) {
            error = "cannot inspect published model directory: " +
                    fs_error.message();
            return false;
        }
        ModelManifest published;
        if (!LoadCachedStep(manifest.model_step, published, error) ||
            !SameWireManifest(published, validated)) {
            if (error.empty()) {
                error = "published model step has a different identity";
            }
            return false;
        }
        std::string discard_error;
        if (!DiscardTemporary(manifest, discard_error)) {
            error = discard_error;
            return false;
        }
        manifest = std::move(published);
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            cached_steps_.insert(manifest.model_step);
        }
        return true;
    }

    fs::rename(temporary_dir, final_dir, fs_error);
    if (fs_error) {
        error = "cannot atomically publish prepared model: " +
                fs_error.message();
        return false;
    }
    if (!FsyncDirectory(cache_root, error)) return false;
    manifest = std::move(validated);
    manifest.model_path = (final_dir / kCachedModelFile).string();
    manifest.manifest_path =
        (final_dir / kModelManifestFile).string();
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cached_steps_.insert(manifest.model_step);
    }
    error.clear();
    return true;
}

bool ModelDistributorClient::DiscardTemporary(
    const ModelManifest& manifest,
    std::string& error) const {
    namespace fs = std::filesystem;
    std::string lineage_id;
    std::string lineage_key;
    fs::path cache_root;
    if (!GetPinnedModelLineage(lineage_id, lineage_key, error) ||
        manifest.model_lineage_id != lineage_id ||
        !OpenActiveLineageNamespace(
            config_, lineage_id, lineage_key, cache_root, error)) {
        if (error.empty()) {
            error = "temporary model does not match the active lineage cache";
        }
        return false;
    }
    const fs::path directory = fs::path(manifest.model_path).parent_path();
    std::error_code fs_error;
    if (!fs::exists(directory, fs_error) && !fs_error) return true;
    if (fs_error || directory.parent_path() != cache_root) {
        error = "refusing to remove a non-temporary model directory";
        return false;
    }
    return RemoveValidatedPrivateDirectory(
        cache_root, directory, ".tmp-", error);
}

bool ModelDistributorClient::GetFirstMissingCachedStep(
    ModelStep floor_model_step,
    ModelStep latest_model_step,
    std::optional<ModelStep>& missing_model_step,
    std::string& error) const {
    missing_model_step.reset();
    if (latest_model_step < floor_model_step) {
        error = "requested cache range is invalid";
        return false;
    }
    std::lock_guard<std::mutex> lock(cache_mutex_);
    for (ModelStep step = floor_model_step;; ++step) {
        if (cached_steps_.find(step) == cached_steps_.end()) {
            missing_model_step = step;
            break;
        }
        if (step == latest_model_step) break;
    }
    error.clear();
    return true;
}

bool ModelDistributorClient::PruneCache(
    const std::set<ModelStep>& protected_steps,
    std::string& error) {
    namespace fs = std::filesystem;
    std::string lineage_id;
    std::string lineage_key;
    fs::path cache_root;
    if (!GetPinnedModelLineage(lineage_id, lineage_key, error) ||
        !OpenActiveLineageNamespace(
            config_, lineage_id, lineage_key, cache_root, error)) {
        return false;
    }
    std::error_code fs_error;
    std::vector<fs::path> private_prune_directories;
    for (fs::directory_iterator iterator(cache_root, fs_error), end;
         !fs_error && iterator != end; iterator.increment(fs_error)) {
        ModelStep step = 0;
        const std::string name = iterator->path().filename().string();
        if (ParsePrivateCacheDirectory(name, ".prune-", step)) {
            private_prune_directories.push_back(iterator->path());
        } else if (IsPrivatePruneDirectory(iterator->path())) {
            error = "model cache contains an invalid prune residue: " + name;
            return false;
        }
    }
    if (fs_error) {
        error = "cannot scan model prune residue: " + fs_error.message();
        return false;
    }
    std::sort(private_prune_directories.begin(),
              private_prune_directories.end());
    for (const auto& directory : private_prune_directories) {
        if (!RemoveValidatedPrivateDirectory(
                cache_root, directory, ".prune-", error)) {
            return false;
        }
    }
    if (!private_prune_directories.empty() &&
        !FsyncDirectory(cache_root, error)) {
        return false;
    }

    std::vector<ModelManifest> models;
    if (!ListCachedModels(models, error)) return false;
    if (models.size() <= kCacheRetentionSteps) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cached_steps_.clear();
        for (const auto& model : models) {
            cached_steps_.insert(model.model_step);
        }
        error.clear();
        return true;
    }

    const std::size_t ordinary_floor_index =
        models.size() - kCacheRetentionSteps;
    for (std::size_t index = 0; index < ordinary_floor_index; ++index) {
        const ModelManifest& model = models[index];
        if (protected_steps.find(model.model_step) != protected_steps.end()) {
            continue;
        }
        const fs::path canonical =
            cache_root / CacheStepDirectoryName(model.model_step);
        ModelManifest validated;
        if (!LoadCachedStep(model.model_step, validated, error) ||
            !SameWireManifest(model, validated)) {
            if (error.empty()) {
                error = "cached model changed before pruning";
            }
            return false;
        }
        const fs::path quarantine = AllocatePrivateCacheDirectory(
            cache_root, ".prune-", model.model_step, error);
        if (quarantine.empty()) return false;
        fs_error.clear();
        fs::remove(quarantine, fs_error);
        if (fs_error) {
            error = "cannot prepare model prune quarantine: " +
                    fs_error.message();
            return false;
        }
        fs::rename(canonical, quarantine, fs_error);
        if (fs_error) {
            error = "cannot atomically quarantine cached model: " +
                    fs_error.message();
            return false;
        }
        if (!FsyncDirectory(cache_root, error) ||
            !RemoveValidatedPrivateDirectory(
                cache_root, quarantine, ".prune-", error) ||
            !FsyncDirectory(cache_root, error)) {
            return false;
        }
    }
    std::vector<ModelManifest> retained_models;
    if (!ListCachedModels(retained_models, error)) return false;
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cached_steps_.clear();
        for (const auto& model : retained_models) {
            cached_steps_.insert(model.model_step);
        }
    }
    error.clear();
    return true;
}
