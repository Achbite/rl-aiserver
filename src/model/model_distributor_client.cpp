#include "model/model_distributor_client.h"

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <thread>
#include <vector>

namespace {

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

bool ParseCanonicalVersionDirectory(const std::string& name,
                                    ModelVersion& version) {
    if (name.size() < 6 ||
        !std::all_of(name.begin(), name.end(), [](unsigned char value) {
            return std::isdigit(value) != 0;
        })) {
        return false;
    }
    try {
        const unsigned long long parsed = std::stoull(name);
        version = static_cast<ModelVersion>(parsed);
    } catch (...) {
        return false;
    }
    std::ostringstream canonical;
    canonical << std::setfill('0') << std::setw(6) << version;
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
                                ModelVersion& version) {
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
    return ParseCanonicalVersionDirectory(
               remainder.substr(0, process_separator), version) &&
           IsUnsignedDecimal(remainder.substr(
               process_separator + 1,
               last_separator - process_separator - 1)) &&
           IsUnsignedDecimal(remainder.substr(last_separator + 1));
}

bool RemoveValidatedPrivateDirectory(
    const std::filesystem::path& cache_root,
    const std::filesystem::path& directory,
    const std::string& prefix,
    std::string& error) {
    namespace fs = std::filesystem;
    ModelVersion ignored_version = 0;
    std::error_code fs_error;
    const auto status = fs::symlink_status(directory, fs_error);
    if (fs_error || fs::is_symlink(status) || !fs::is_directory(status) ||
        directory.parent_path() != cache_root ||
        !ParsePrivateCacheDirectory(
            directory.filename().string(), prefix, ignored_version)) {
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
    ModelVersion model_version,
    std::string& error) {
    namespace fs = std::filesystem;
    static std::atomic<uint64_t> next_private_id{1};
    std::error_code fs_error;
    for (int attempt = 0; attempt < 100; ++attempt) {
        const fs::path candidate =
            cache_root /
            (prefix + ModelDistributorClient::CacheVersionDirectoryName(
                          model_version) +
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
    std::optional<ModelVersion> expected_version,
    std::string& error) const {
    return ValidateModelManifest(
        config_, source, expected_version, error);
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
    const fs::path cache_root =
        fs::path(config_.model.local_train_dir) / "cache";
    std::error_code fs_error;
    fs::create_directories(cache_root, fs_error);
    if (fs_error) {
        error = "cannot create model cache directory: " +
                fs_error.message();
        return false;
    }
    const auto cache_status = fs::symlink_status(cache_root, fs_error);
    if (fs_error || fs::is_symlink(cache_status) ||
        !fs::is_directory(cache_status)) {
        error = "model cache root must be a real directory";
        return false;
    }

    const fs::path temporary_dir = AllocatePrivateCacheDirectory(
        cache_root, ".tmp-",
        source.identity().model_version(), error);
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
        temporary_dir / config_.model.manifest_name;
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
                                   ModelVersion model_version,
                                   bool latest,
                                   ModelManifest& manifest,
    std::string& error) {
    training::GetModelManifestReq request;
    request.mutable_requested_model()->set_model_lineage_id(
        config_.model.expected_model_lineage_id);
    if (!latest) {
        request.mutable_requested_model()->set_model_version(model_version);
    }
    request.mutable_requester()->CopyFrom(requester_identity_);
    request.set_latest_in_lineage(latest);
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
            source,
            latest ? std::nullopt
                   : std::optional<ModelVersion>(model_version),
            error)) {
        return false;
    }

    std::string local_path;
    if (!DownloadToTemporary(source, aiserver_id, local_path, error)) {
        return false;
    }
    const std::filesystem::path manifest_path =
        std::filesystem::path(local_path).parent_path() /
        config_.model.manifest_name;
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

bool ModelDistributorClient::FetchVersion(
    const std::string& aiserver_id,
    ModelVersion model_version,
    ModelManifest& manifest,
    std::string& error) {
    return Fetch(
        aiserver_id, model_version, false, manifest, error);
}

bool ModelDistributorClient::GetLatestIdentity(
    const std::string& aiserver_id,
    ModelVersion& model_version,
    std::string& checksum,
    std::string& error) {
    training::GetModelManifestReq request;
    request.mutable_requested_model()->set_model_lineage_id(
        config_.model.expected_model_lineage_id);
    request.mutable_requester()->CopyFrom(requester_identity_);
    request.set_latest_in_lineage(true);
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
    if (!ValidateManifest(source, std::nullopt, error)) {
        return false;
    }
    model_version = source.identity().model_version();
    checksum = source.identity().artifact_digest().hex();
    return true;
}

bool ModelDistributorClient::GetAvailableRange(
    const std::string& aiserver_id,
    AvailableRange& range,
    std::string& error) {
    range = AvailableRange{};
    training::GetModelManifestReq request;
    request.mutable_requested_model()->set_model_lineage_id(
        config_.model.expected_model_lineage_id);
    request.mutable_requester()->CopyFrom(requester_identity_);
    request.set_latest_in_lineage(true);
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
    const auto& latest = response.manifest();
    if (!ValidateManifest(latest, std::nullopt, error)) return false;

    if (!response.has_available_floor_model_version() ||
        !response.has_latest_available_model_version()) {
        error = "Model Distributor available range is missing";
        return false;
    }

    const uint64_t floor_version =
        response.available_floor_model_version();
    const uint64_t latest_version =
        response.latest_available_model_version();
    if (floor_version > latest_version ||
        latest_version != latest.identity().model_version()) {
        error = "Model Distributor available range is invalid";
        return false;
    }
    range.floor_model_version = floor_version;
    range.latest_model_version = latest_version;
    range.latest_checksum = latest.identity().artifact_digest().hex();
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
        requester_identity_.instance_id() + "-load-v" +
        std::to_string(manifest.model_version));
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

std::string ModelDistributorClient::CacheVersionDirectoryName(
    ModelVersion model_version) {
    std::ostringstream output;
    output << std::setfill('0') << std::setw(6) << model_version;
    return output.str();
}

bool ModelDistributorClient::LoadCachedVersion(
    ModelVersion model_version,
    ModelManifest& manifest,
    std::string& error) const {
    namespace fs = std::filesystem;
    const fs::path directory =
        fs::path(config_.model.local_train_dir) / "cache" /
        CacheVersionDirectoryName(model_version);
    std::error_code fs_error;
    const auto directory_status = fs::symlink_status(directory, fs_error);
    if (fs_error || fs::is_symlink(directory_status) ||
        !fs::is_directory(directory_status)) {
        error = "cached model version directory is invalid";
        return false;
    }
    std::set<std::string> entries;
    for (fs::directory_iterator iterator(directory, fs_error), end;
         !fs_error && iterator != end; iterator.increment(fs_error)) {
        const auto entry_status =
            fs::symlink_status(iterator->path(), fs_error);
        if (fs_error || fs::is_symlink(entry_status) ||
            !fs::is_regular_file(entry_status)) {
            error = "cached model version contains an invalid entry";
            return false;
        }
        entries.insert(iterator->path().filename().string());
    }
    const std::set<std::string> expected_entries{
        kCachedModelFile, config_.model.manifest_name};
    if (fs_error || entries != expected_entries) {
        error = "cached model version is not a complete two-file artifact";
        return false;
    }
    if (!LoadModelManifestFile(
            config_,
            (directory / config_.model.manifest_name).string(),
            manifest, error)) {
        return false;
    }
    if (manifest.model_version != model_version ||
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
    const fs::path cache_root =
        fs::path(config_.model.local_train_dir) / "cache";
    std::error_code fs_error;
    fs::create_directories(cache_root, fs_error);
    if (fs_error) {
        error = "cannot create model cache directory: " + fs_error.message();
        return false;
    }
    const auto root_status = fs::symlink_status(cache_root, fs_error);
    if (fs_error || fs::is_symlink(root_status) ||
        !fs::is_directory(root_status)) {
        error = "model cache root must be a real directory";
        return false;
    }
    for (fs::directory_iterator iterator(cache_root, fs_error), end;
         !fs_error && iterator != end; iterator.increment(fs_error)) {
        const fs::path path = iterator->path();
        const auto status = fs::symlink_status(path, fs_error);
        if (fs_error) break;
        ModelVersion directory_version = 0;
        if (fs::is_symlink(status) || !fs::is_directory(status) ||
            !ParseCanonicalVersionDirectory(
                path.filename().string(), directory_version)) {
            error = "model cache contains an unrecognized entry: " +
                    path.filename().string();
            models.clear();
            return false;
        }
        ModelManifest candidate;
        std::string validation_error;
        if (!LoadCachedVersion(
                directory_version, candidate, validation_error)) {
            error = "cached model version is invalid: " +
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
        return left.model_version < right.model_version;
    });
    error.clear();
    return true;
}

bool ModelDistributorClient::RecoverCache(
    std::vector<ModelManifest>& models,
    std::string& error) {
    namespace fs = std::filesystem;
    models.clear();
    const fs::path cache_root =
        fs::path(config_.model.local_train_dir) / "cache";
    std::error_code fs_error;
    fs::create_directories(cache_root, fs_error);
    if (fs_error) {
        error = "cannot create model cache directory: " + fs_error.message();
        return false;
    }
    const auto root_status = fs::symlink_status(cache_root, fs_error);
    if (fs_error || fs::is_symlink(root_status) ||
        !fs::is_directory(root_status)) {
        error = "model cache root must be a real directory";
        return false;
    }

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
        ModelVersion private_version = 0;
        if (ParsePrivateCacheDirectory(name, ".tmp-", private_version)) {
            if (!RemoveValidatedPrivateDirectory(
                    cache_root, path, ".tmp-", error)) {
                return false;
            }
            changed = true;
            continue;
        }
        if (ParsePrivateCacheDirectory(name, ".prune-", private_version)) {
            if (!RemoveValidatedPrivateDirectory(
                    cache_root, path, ".prune-", error)) {
                return false;
            }
            changed = true;
            continue;
        }

        ModelVersion directory_version = 0;
        const auto status = fs::symlink_status(path, fs_error);
        if (fs_error || fs::is_symlink(status) ||
            !fs::is_directory(status) ||
            !ParseCanonicalVersionDirectory(name, directory_version)) {
            error = "model cache contains an unrecognized entry: " + name;
            return false;
        }
        ModelManifest candidate;
        std::string validation_error;
        if (!LoadCachedVersion(
                directory_version, candidate, validation_error)) {
            const fs::path quarantine = AllocatePrivateCacheDirectory(
                cache_root, ".prune-", directory_version, error);
            if (quarantine.empty()) return false;
            fs::remove(quarantine, fs_error);
            if (fs_error) {
                error = "cannot prepare corrupt cache quarantine: " +
                        fs_error.message();
                return false;
            }
            fs::rename(path, quarantine, fs_error);
            if (fs_error || !RemoveValidatedPrivateDirectory(
                                cache_root, quarantine, ".prune-", error)) {
                if (error.empty()) {
                    error = "cannot remove corrupt cached model: " +
                            fs_error.message();
                }
                return false;
            }
            changed = true;
        }
    }
    if (changed && !FsyncDirectory(cache_root, error)) return false;
    if (!ListCachedModels(models, error)) return false;
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cached_versions_.clear();
        for (const auto& model : models) {
            cached_versions_.insert(model.model_version);
        }
    }
    if (!PruneCache(error) || !ListCachedModels(models, error)) return false;
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
    const fs::path temporary_model = manifest.model_path;
    const fs::path temporary_dir = temporary_model.parent_path();
    const fs::path cache_root =
        fs::path(config_.model.local_train_dir) / "cache";
    const fs::path final_dir =
        cache_root / CacheVersionDirectoryName(manifest.model_version);
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
            (temporary_dir / config_.model.manifest_name).string(),
            validated, error) ||
        !SameWireManifest(validated, manifest) ||
        validated.model_version != manifest.model_version) {
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
        if (!LoadCachedVersion(manifest.model_version, published, error) ||
            !SameWireManifest(published, validated)) {
            if (error.empty()) {
                error = "published model version has a different identity";
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
            cached_versions_.insert(manifest.model_version);
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
        (final_dir / config_.model.manifest_name).string();
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cached_versions_.insert(manifest.model_version);
    }
    error.clear();
    return true;
}

bool ModelDistributorClient::DiscardTemporary(
    const ModelManifest& manifest,
    std::string& error) const {
    namespace fs = std::filesystem;
    const fs::path directory = fs::path(manifest.model_path).parent_path();
    const fs::path cache_root =
        fs::path(config_.model.local_train_dir) / "cache";
    std::error_code fs_error;
    if (!fs::exists(directory, fs_error) && !fs_error) return true;
    if (fs_error || directory.parent_path() != cache_root) {
        error = "refusing to remove a non-temporary model directory";
        return false;
    }
    return RemoveValidatedPrivateDirectory(
        cache_root, directory, ".tmp-", error);
}

bool ModelDistributorClient::GetFirstMissingCachedVersion(
    ModelVersion floor_model_version,
    ModelVersion latest_model_version,
    std::optional<ModelVersion>& missing_model_version,
    std::string& error) const {
    missing_model_version.reset();
    if (latest_model_version < floor_model_version) {
        error = "requested cache range is invalid";
        return false;
    }
    std::lock_guard<std::mutex> lock(cache_mutex_);
    for (ModelVersion version = floor_model_version;; ++version) {
        if (cached_versions_.find(version) == cached_versions_.end()) {
            missing_model_version = version;
            break;
        }
        if (version == latest_model_version) break;
    }
    error.clear();
    return true;
}

bool ModelDistributorClient::PruneCache(std::string& error) {
    namespace fs = std::filesystem;
    const fs::path cache_root =
        fs::path(config_.model.local_train_dir) / "cache";
    std::error_code fs_error;
    std::vector<fs::path> private_prune_directories;
    for (fs::directory_iterator iterator(cache_root, fs_error), end;
         !fs_error && iterator != end; iterator.increment(fs_error)) {
        ModelVersion version = 0;
        const std::string name = iterator->path().filename().string();
        if (ParsePrivateCacheDirectory(name, ".prune-", version)) {
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
    if (models.size() <= kCacheRetentionVersions) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cached_versions_.clear();
        for (const auto& model : models) {
            cached_versions_.insert(model.model_version);
        }
        error.clear();
        return true;
    }

    const std::size_t remove_count =
        models.size() - kCacheRetentionVersions;
    for (std::size_t index = 0; index < remove_count; ++index) {
        const ModelManifest& model = models[index];
        const fs::path canonical =
            cache_root / CacheVersionDirectoryName(model.model_version);
        ModelManifest validated;
        if (!LoadCachedVersion(model.model_version, validated, error) ||
            !SameWireManifest(model, validated)) {
            if (error.empty()) {
                error = "cached model changed before pruning";
            }
            return false;
        }
        const fs::path quarantine = AllocatePrivateCacheDirectory(
            cache_root, ".prune-", model.model_version, error);
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
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cached_versions_.clear();
        for (std::size_t index = remove_count; index < models.size(); ++index) {
            cached_versions_.insert(models[index].model_version);
        }
    }
    error.clear();
    return true;
}
