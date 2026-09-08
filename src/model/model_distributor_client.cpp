#include "model/model_distributor_client.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <thread>

namespace {

constexpr char kCacheDirectory[] = "cache";
constexpr char kLineagesDirectory[] = "lineages";

std::string EncodeLineageKey(const std::string& value) {
    if (value.empty()) return "";
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const unsigned char character : value) {
        output << std::setw(2) << static_cast<unsigned int>(character);
    }
    return output.str();
}

bool ValidModelDistributorAuthority(
    const common::ServiceInstanceIdentity& identity) {
    return !identity.component().empty() && !identity.instance_id().empty() &&
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

bool EnsureCacheRoot(const ModelConfig& config,
                     std::filesystem::path& cache_root,
                     std::string& error) {
    namespace fs = std::filesystem;
    cache_root =
        fs::path(config.local_train_dir) / kCacheDirectory;
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

bool OpenOrCreateLineageCache(
    const ModelConfig& config,
    const std::string& lineage_id,
    const std::string& lineage_key,
    std::filesystem::path& active_root,
    std::string& error) {
    namespace fs = std::filesystem;
    if (lineage_id.empty() || EncodeLineageKey(lineage_id) != lineage_key) {
        error = "selected model lineage identity is invalid";
        return false;
    }
    fs::path cache_root;
    if (!EnsureCacheRoot(config, cache_root, error)) return false;
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
    if (created_active && !FsyncDirectory(lineages_root, error)) {
        return false;
    }
    if (created_lineages && !FsyncDirectory(cache_root, error)) {
        return false;
    }
    error.clear();
    return true;
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
    const ModelDistributionConfig& config,
    const ModelConfig& model_config,
    std::string producer_instance_id,
    uint64_t producer_lifecycle_epoch)
    : config_(config), model_config_(model_config) {
    requester_identity_.set_component("rl-aiserver");
    requester_identity_.set_instance_id(std::move(producer_instance_id));
    requester_identity_.set_lifecycle_epoch(producer_lifecycle_epoch);
    const std::string address =
        config_.host + ":" + std::to_string(config_.port);
    channel_ = grpc::CreateChannel(
        address, grpc::InsecureChannelCredentials());
    stub_ = training::ModelDistributorService::NewStub(channel_);
}

bool ModelDistributorClient::ValidateManifest(
    const training::ModelArtifactManifest& source,
    std::optional<ModelStep> expected_step,
    std::string& error) const {
    if (!ValidateModelManifest(source, expected_step, error)) {
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
    const std::string lineage_key = EncodeLineageKey(lineage_id);
    if (lineage_key.empty()) {
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
    if (lineage_id.empty() || EncodeLineageKey(lineage_id) != lineage_key) {
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
    std::string lineage_id;
    std::string lineage_key;
    fs::path cache_root;
    if (!GetPinnedModelLineage(lineage_id, lineage_key, error) ||
        source.identity().model_lineage_id() != lineage_id ||
        !OpenOrCreateLineageCache(
            model_config_, lineage_id, lineage_key, cache_root, error)) {
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
        std::chrono::milliseconds(config_.rpc_timeout_ms));
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
    const fs::path manifest_path =
        temporary_dir / kModelManifestFile;
    if (!WriteModelManifestFile(source, manifest_path.string(), error) ||
        !FsyncDirectory(temporary_dir, error)) {
        fs::remove_all(temporary_dir, fs_error);
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
        std::chrono::milliseconds(config_.rpc_timeout_ms));
    const grpc::Status status =
        stub_->GetModelManifest(&context, request, &response);
    if (!status.ok() ||
        response.result() != training::MODEL_LOOKUP_RESULT_FOUND ||
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
    AssignModelManifest(source, local_path, manifest);
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
        std::chrono::milliseconds(config_.rpc_timeout_ms));
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
    if (!response.has_latest_model() ||
        response.latest_model().model_lineage_id().empty() ||
        !response.latest_model().has_model_step()) {
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
        std::chrono::milliseconds(config_.rpc_timeout_ms));
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
        std::to_string(manifest.model_step()));
    request.set_load_status(load_status);
    request.set_message(message);
    constexpr int kMaxAttempts = 3;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        training::AckModelRsp response;
        grpc::ClientContext context;
        context.set_deadline(
            std::chrono::system_clock::now() +
            std::chrono::milliseconds(
                config_.rpc_timeout_ms));
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
        !OpenOrCreateLineageCache(
            model_config_, lineage_id, lineage_key, cache_root, error)) {
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
            (directory / kModelManifestFile).string(),
            manifest, error)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(lineage_mutex_);
        if (pinned_model_lineage_id_.has_value() &&
            manifest.model_lineage_id() != *pinned_model_lineage_id_) {
            error = "cached model does not match the discovered training lineage";
            return false;
        }
    }
    if (manifest.model_step() != model_step ||
        fs::path(manifest.model_path).filename() != kCachedModelFile) {
        error = "cached model directory identity mismatch";
        return false;
    }
    error.clear();
    return true;
}

bool ModelDistributorClient::PublishPrepared(
    ModelManifest& manifest,
    std::string& error) {
    namespace fs = std::filesystem;
    if (fs::path(manifest.model_path).filename() != kCachedModelFile) {
        error = "prepared model identity is invalid";
        return false;
    }
    std::string lineage_id;
    std::string lineage_key;
    fs::path cache_root;
    if (!GetPinnedModelLineage(lineage_id, lineage_key, error) ||
        manifest.model_lineage_id() != lineage_id ||
        !OpenOrCreateLineageCache(
            model_config_, lineage_id, lineage_key, cache_root, error)) {
        if (error.empty()) {
            error = "prepared model does not match the active lineage cache";
        }
        return false;
    }
    const fs::path temporary_model = manifest.model_path;
    const fs::path temporary_dir = temporary_model.parent_path();
    const fs::path final_dir =
        cache_root / CacheStepDirectoryName(manifest.model_step());
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
            (temporary_dir / kModelManifestFile).string(),
            validated, error) ||
        !SameWireManifest(validated, manifest) ||
        validated.model_step() != manifest.model_step()) {
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
        if (!LoadCachedStep(manifest.model_step(), published, error) ||
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
            cached_steps_.insert(manifest.model_step());
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
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cached_steps_.insert(manifest.model_step());
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
        manifest.model_lineage_id() != lineage_id ||
        !OpenOrCreateLineageCache(
            model_config_, lineage_id, lineage_key, cache_root, error)) {
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

bool ModelDistributorClient::PruneCache(
    const std::set<ModelStep>& protected_steps,
    std::string& error) {
    namespace fs = std::filesystem;
    std::string lineage_id;
    std::string lineage_key;
    fs::path cache_root;
    if (!GetPinnedModelLineage(lineage_id, lineage_key, error) ||
        !OpenOrCreateLineageCache(
            model_config_, lineage_id, lineage_key, cache_root, error)) {
        return false;
    }
    std::set<ModelStep> cached_steps;
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cached_steps = cached_steps_;
    }
    for (ModelStep model_step : cached_steps) {
        if (protected_steps.find(model_step) != protected_steps.end()) continue;
        const fs::path canonical =
            cache_root / CacheStepDirectoryName(model_step);
        ModelManifest validated;
        if (!LoadCachedStep(model_step, validated, error)) return false;
        const fs::path quarantine = AllocatePrivateCacheDirectory(
            cache_root, ".prune-", model_step, error);
        if (quarantine.empty()) return false;
        std::error_code fs_error;
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
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cached_steps_.erase(model_step);
    }
    error.clear();
    return true;
}
