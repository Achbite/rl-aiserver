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

ModelDistributorClient::ModelDistributorClient(
    const ModelDistributionConfig& distribution,
    const ModelConfig& model)
    : distribution_(distribution), model_(model) {
    const std::string address =
        distribution_.host + ":" + std::to_string(distribution_.port);
    channel_ = grpc::CreateChannel(
        address, grpc::InsecureChannelCredentials());
    stub_ = maze::ModelDistributorService::NewStub(channel_);
}

bool ModelDistributorClient::ValidateManifest(
    const maze::ModelArtifactManifest& source,
    int expected_version,
    std::string& error) const {
    if (source.schema_version() != 1 ||
        source.contract_version() != distribution_.contract_version ||
        !source.ready()) {
        error = "model manifest version or ready state is invalid";
        return false;
    }
    if ((expected_version >= 0 &&
         source.model_version() != expected_version) ||
        source.size_bytes() <= 0 || !IsSha256(source.sha256())) {
        error = "model manifest identity is invalid";
        return false;
    }
    if (!ShapeEquals(
            source.input_shape(), {1, model_.expected_obs_dim}) ||
        !ShapeEquals(
            source.action_shape(), {1, model_.expected_action_dim}) ||
        !ShapeEquals(source.value_shape(), {1, 1})) {
        error = "model manifest shape does not match AIServer";
        return false;
    }
    if (source.model_file().empty() ||
        std::filesystem::path(source.model_file()).filename() !=
            std::filesystem::path(source.model_file())) {
        error = "model_file must be a file name";
        return false;
    }
    return true;
}

bool ModelDistributorClient::Download(
    const maze::ModelArtifactManifest& source,
    const std::string& aiserver_id,
    std::string& local_path,
    std::string& error) {
    namespace fs = std::filesystem;
    const fs::path run_dir = fs::path(model_.p2p_dir) / source.run_id();
    std::error_code fs_error;
    fs::create_directories(run_dir, fs_error);
    if (fs_error) {
        error = "cannot create model cache: " + fs_error.message();
        return false;
    }
    const fs::path final_path = run_dir / source.model_file();
    const fs::path temporary =
        final_path.string() + ".tmp." + std::to_string(::getpid());
    const int descriptor = ::open(
        temporary.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (descriptor < 0) {
        error = "cannot open model download temporary file";
        return false;
    }

    maze::DownloadModelReq request;
    request.set_run_id(source.run_id());
    request.set_model_version(source.model_version());
    request.set_aiserver_id(aiserver_id);
    grpc::ClientContext context;
    context.set_deadline(
        std::chrono::system_clock::now() +
        std::chrono::milliseconds(distribution_.rpc_timeout_ms));
    std::unique_ptr<grpc::ClientReader<maze::ModelChunk>> reader =
        stub_->DownloadModel(&context, request);
    int64_t expected_offset = 0;
    maze::ModelChunk chunk;
    bool write_ok = true;
    while (reader->Read(&chunk)) {
        if (chunk.run_id() != source.run_id() ||
            chunk.model_version() != source.model_version() ||
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
    if (checksum != source.sha256()) {
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

bool ModelDistributorClient::Fetch(const std::string& run_id,
                                   const std::string& aiserver_id,
                                   int model_version,
                                   bool latest,
                                   ModelManifest& manifest,
                                   std::string& error) {
    maze::GetModelManifestReq request;
    request.set_run_id(run_id);
    request.set_model_version(model_version);
    request.set_aiserver_id(aiserver_id);
    request.set_latest(latest);
    maze::GetModelManifestRsp response;
    grpc::ClientContext context;
    context.set_deadline(
        std::chrono::system_clock::now() +
        std::chrono::milliseconds(distribution_.rpc_timeout_ms));
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
    if (source.run_id() != run_id ||
        !ValidateManifest(
            source, latest ? -1 : model_version, error)) {
        if (error.empty()) error = "model manifest run_id mismatch";
        return false;
    }

    std::string local_path;
    if (!Download(source, aiserver_id, local_path, error)) return false;
    manifest.schema_version = static_cast<int>(source.schema_version());
    manifest.contract_version = source.contract_version();
    manifest.run_id = source.run_id();
    manifest.model_version = source.model_version();
    manifest.artifact_uri = source.artifact_uri();
    manifest.model_file = source.model_file();
    manifest.size_bytes = source.size_bytes();
    manifest.sha256 = source.sha256();
    manifest.input_shape.assign(
        source.input_shape().begin(), source.input_shape().end());
    manifest.action_shape.assign(
        source.action_shape().begin(), source.action_shape().end());
    manifest.value_shape.assign(
        source.value_shape().begin(), source.value_shape().end());
    manifest.seed = source.seed();
    manifest.ready = source.ready();
    manifest.published_ts_ms = source.published_ts_ms();
    manifest.model_path = local_path;
    return true;
}

bool ModelDistributorClient::FetchLatest(
    const std::string& run_id,
    const std::string& aiserver_id,
    ModelManifest& manifest,
    std::string& error) {
    return Fetch(run_id, aiserver_id, -1, true, manifest, error);
}

bool ModelDistributorClient::FetchVersion(
    const std::string& run_id,
    const std::string& aiserver_id,
    int model_version,
    ModelManifest& manifest,
    std::string& error) {
    return Fetch(
        run_id, aiserver_id, model_version, false, manifest, error);
}

bool ModelDistributorClient::GetLatestIdentity(
    const std::string& run_id,
    const std::string& aiserver_id,
    int& model_version,
    std::string& checksum,
    std::string& error) {
    maze::GetModelManifestReq request;
    request.set_run_id(run_id);
    request.set_aiserver_id(aiserver_id);
    request.set_latest(true);
    maze::GetModelManifestRsp response;
    grpc::ClientContext context;
    context.set_deadline(
        std::chrono::system_clock::now() +
        std::chrono::milliseconds(distribution_.rpc_timeout_ms));
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
    if (source.run_id() != run_id ||
        !ValidateManifest(source, -1, error)) {
        if (error.empty()) error = "model manifest run_id mismatch";
        return false;
    }
    model_version = source.model_version();
    checksum = source.sha256();
    return true;
}

bool ModelDistributorClient::Ack(const ModelManifest& manifest,
                                 const std::string& run_id,
                                 const std::string& aiserver_id,
                                 maze::ModelLoadStatus load_status,
                                 const std::string& message,
                                 std::string& error) {
    maze::AckModelReq request;
    request.set_run_id(run_id);
    request.set_aiserver_id(aiserver_id);
    request.set_model_version(manifest.model_version);
    request.set_sha256(manifest.sha256);
    request.set_load_status(load_status);
    request.set_message(message);
    maze::AckModelRsp response;
    grpc::ClientContext context;
    context.set_deadline(
        std::chrono::system_clock::now() +
        std::chrono::milliseconds(distribution_.rpc_timeout_ms));
    const grpc::Status status = stub_->AckModel(&context, request, &response);
    if (!status.ok() ||
        (response.result() != maze::MODEL_ACK_RESULT_APPLIED &&
         response.result() != maze::MODEL_ACK_RESULT_ALREADY_APPLIED)) {
        error = status.ok()
                    ? response.message()
                    : "model ACK RPC failed: " + status.error_message();
        return false;
    }
    return true;
}
