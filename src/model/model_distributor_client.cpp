#include "model/model_distributor_client.h"

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <initializer_list>

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

}  // namespace

ModelDistributorClient::ModelDistributorClient(const AIServerConfig& config)
    : config_(config) {
    const std::string address =
        config_.model_distribution.host + ":" +
        std::to_string(config_.model_distribution.port);
    channel_ = grpc::CreateChannel(
        address, grpc::InsecureChannelCredentials());
    stub_ = training::ModelDistributorService::NewStub(channel_);
}

bool ModelDistributorClient::ValidateManifest(
    const training::ModelArtifactManifest& source,
    int expected_version,
    std::string& error) const {
    return ValidateModelManifest(
        config_, source, expected_version, error);
}

bool ModelDistributorClient::Download(
    const training::ModelArtifactManifest& source,
    const std::string& aiserver_id,
    std::string& local_path,
    std::string& error) {
    namespace fs = std::filesystem;
    const fs::path incoming_dir =
        fs::path(config_.model.local_train_dir) / "incoming";
    std::error_code fs_error;
    fs::create_directories(incoming_dir, fs_error);
    if (fs_error) {
        error = "cannot create incoming model directory: " +
                fs_error.message();
        return false;
    }
    const fs::path final_path = incoming_dir / source.model_file();
    const fs::path temporary =
        final_path.string() + ".tmp." + std::to_string(::getpid());
    const int descriptor = ::open(
        temporary.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (descriptor < 0) {
        error = "cannot open model download temporary file";
        return false;
    }

    training::DownloadModelReq request;
    *request.mutable_requested_model() = source.identity();
    request.mutable_requester()->set_component("aiserver");
    request.mutable_requester()->set_instance_id(aiserver_id);
    request.mutable_requester()->set_lifecycle_epoch(1);
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
        fs::remove(temporary, fs_error);
        error = status.ok()
                    ? "model stream offset or size mismatch"
                    : "model download failed: " + status.error_message();
        return false;
    }
    std::string checksum;
    if (!ComputeFileSha256(temporary.string(), checksum, error)) {
        fs::remove(temporary, fs_error);
        return false;
    }
    if (checksum != source.identity().artifact_digest().hex()) {
        fs::remove(temporary, fs_error);
        error = "downloaded model checksum mismatch";
        return false;
    }
    fs::rename(temporary, final_path, fs_error);
    if (fs_error) {
        fs::remove(temporary, fs_error);
        error = "cannot install downloaded model: " + fs_error.message();
        return false;
    }
    local_path = final_path.string();
    return true;
}

bool ModelDistributorClient::Fetch(const std::string& aiserver_id,
                                   int model_version,
                                   bool latest,
                                   ModelManifest& manifest,
    std::string& error) {
    training::GetModelManifestReq request;
    request.mutable_requested_model()->set_model_lineage_id(
        config_.model.expected_model_lineage_id);
    if (!latest) {
        request.mutable_requested_model()->set_model_version(
            static_cast<uint64_t>(model_version));
    }
    request.mutable_requester()->set_component("aiserver");
    request.mutable_requester()->set_instance_id(aiserver_id);
    request.mutable_requester()->set_lifecycle_epoch(1);
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
    if (!ValidateManifest(source, model_version, error)) {
        return false;
    }

    std::string local_path;
    if (!Download(source, aiserver_id, local_path, error)) return false;
    AssignModelManifest(source, local_path, manifest);
    return true;
}

bool ModelDistributorClient::FetchLatest(
    const std::string& aiserver_id,
    ModelManifest& manifest,
    std::string& error) {
    return Fetch(aiserver_id, -1, true, manifest, error);
}

bool ModelDistributorClient::FetchVersion(
    const std::string& aiserver_id,
    int model_version,
    ModelManifest& manifest,
    std::string& error) {
    return Fetch(
        aiserver_id, model_version, false, manifest, error);
}

bool ModelDistributorClient::GetLatestIdentity(
    const std::string& aiserver_id,
    int& model_version,
    std::string& checksum,
    std::string& error) {
    training::GetModelManifestReq request;
    request.mutable_requested_model()->set_model_lineage_id(
        config_.model.expected_model_lineage_id);
    request.mutable_requester()->set_component("aiserver");
    request.mutable_requester()->set_instance_id(aiserver_id);
    request.mutable_requester()->set_lifecycle_epoch(1);
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
    if (!ValidateManifest(source, -1, error)) {
        return false;
    }
    model_version = static_cast<int>(source.identity().model_version());
    checksum = source.identity().artifact_digest().hex();
    return true;
}

bool ModelDistributorClient::Ack(const ModelManifest& manifest,
                                 const std::string& aiserver_id,
                                 training::ModelLoadStatus load_status,
                                 const std::string& message,
                                 std::string& error) {
    training::AckModelReq request;
    request.mutable_aiserver()->set_component("aiserver");
    request.mutable_aiserver()->set_instance_id(aiserver_id);
    request.mutable_aiserver()->set_lifecycle_epoch(1);
    *request.mutable_model() = manifest.wire.identity();
    request.set_load_instance_id(
        aiserver_id + "-load-v" + std::to_string(manifest.model_version));
    request.set_load_status(load_status);
    request.set_message(message);
    training::AckModelRsp response;
    grpc::ClientContext context;
    context.set_deadline(
        std::chrono::system_clock::now() +
        std::chrono::milliseconds(config_.model_distribution.rpc_timeout_ms));
    const grpc::Status status = stub_->AckModel(&context, request, &response);
    if (!status.ok() ||
        (response.result() != training::MODEL_ACK_RESULT_APPLIED &&
         response.result() != training::MODEL_ACK_RESULT_ALREADY_APPLIED)) {
        error = status.ok()
                    ? response.message()
                    : "model ACK RPC failed: " + status.error_message();
        return false;
    }
    return true;
}

bool ModelDistributorClient::Promote(
    ModelManifest& manifest,
    std::string& previous_path,
    std::string& error) {
    namespace fs = std::filesystem;
    const fs::path source = manifest.model_path;
    const fs::path root = config_.model.local_train_dir;
    const fs::path active_dir = root / "active";
    const fs::path previous_dir = root / "previous";
    const fs::path active_path = active_dir / "model.onnx";
    const fs::path previous_model = previous_dir / "model.onnx";
    std::error_code fs_error;

    if (!fs::is_regular_file(source, fs_error) || fs_error) {
        error = "incoming model does not exist";
        return false;
    }
    fs::create_directories(active_dir, fs_error);
    if (fs_error) {
        error = "cannot create active model directory: " +
                fs_error.message();
        return false;
    }
    fs::create_directories(previous_dir, fs_error);
    if (fs_error) {
        error = "cannot create previous model directory: " +
                fs_error.message();
        return false;
    }

    fs::remove(previous_model, fs_error);
    fs_error.clear();
    previous_path.clear();
    if (fs::exists(active_path, fs_error)) {
        fs::rename(active_path, previous_model, fs_error);
        if (fs_error) {
            error = "cannot preserve previous model: " +
                    fs_error.message();
            return false;
        }
        previous_path = previous_model.string();
    }

    fs::rename(source, active_path, fs_error);
    if (fs_error) {
        if (!previous_path.empty()) {
            std::error_code rollback_error;
            fs::rename(previous_model, active_path, rollback_error);
        }
        error = "cannot promote incoming model: " + fs_error.message();
        return false;
    }
    manifest.model_path = active_path.string();
    return true;
}
