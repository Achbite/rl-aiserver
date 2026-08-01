#include "maze.grpc.pb.h"
#include "model/model_distributor_client.h"
#include "model/model_manifest.h"

#include <grpcpp/grpcpp.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>

namespace {

class CorruptModelService final
    : public maze::ModelDistributorService::Service {
public:
    explicit CorruptModelService(std::string checksum)
        : checksum_(std::move(checksum)) {}

    void SetWrongShape(bool enabled) {
        wrong_shape_ = enabled;
    }

    grpc::Status GetModelManifest(
        grpc::ServerContext*,
        const maze::GetModelManifestReq* request,
        maze::GetModelManifestRsp* response) override {
        response->set_ret_code(0);
        auto* manifest = response->mutable_manifest();
        manifest->set_schema_version(1);
        manifest->set_contract_version("0.6.0");
        manifest->set_model_version(0);
        manifest->set_artifact_uri("file:///models/model_v000000.onnx");
        manifest->set_model_file("model_v000000.onnx");
        manifest->set_size_bytes(4);
        manifest->set_sha256(checksum_);
        manifest->add_input_shape(1);
        manifest->add_input_shape(wrong_shape_ ? 12 : 13);
        manifest->add_action_shape(1);
        manifest->add_action_shape(9);
        manifest->add_value_shape(1);
        manifest->add_value_shape(1);
        manifest->set_seed(0);
        manifest->set_ready(true);
        return grpc::Status::OK;
    }

    grpc::Status DownloadModel(
        grpc::ServerContext*,
        const maze::DownloadModelReq* request,
        grpc::ServerWriter<maze::ModelChunk>* writer) override {
        maze::ModelChunk chunk;
        chunk.set_model_version(request->model_version());
        chunk.set_offset(0);
        chunk.set_data("evil");
        writer->Write(chunk);
        return grpc::Status::OK;
    }

private:
    std::string checksum_;
    bool wrong_shape_ = false;
};

int Fail(const std::string& message) {
    std::cerr << message << std::endl;
    return 1;
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
                          ("model-client-test-" +
                           std::to_string(::getpid()));
    fs::create_directories(root);
    const fs::path expected_file = root / "expected.onnx";
    {
        std::ofstream output(expected_file, std::ios::binary);
        output << "good";
    }
    std::string checksum;
    std::string checksum_error;
    if (!ComputeFileSha256(
            expected_file.string(), checksum, checksum_error)) {
        fs::remove_all(root);
        return Fail(checksum_error);
    }

    CorruptModelService service(checksum);
    int selected_port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(
        "127.0.0.1:0",
        grpc::InsecureServerCredentials(),
        &selected_port);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (!server || selected_port <= 0) {
        fs::remove_all(root);
        return Fail("failed to start corrupt model service");
    }

    ModelDistributionConfig distribution;
    distribution.host = "127.0.0.1";
    distribution.port = selected_port;
    distribution.rpc_timeout_ms = 1000;
    distribution.contract_version = "0.6.0";
    ModelConfig model;
    model.local_train_dir = (root / "local-train").string();
    ModelDistributorClient client(distribution, model);
    ModelManifest manifest;
    std::string error;
    int latest_version = -1;
    std::string latest_checksum;
    if (!client.GetLatestIdentity(
            "aiserver-0",
            latest_version, latest_checksum, error) ||
        latest_version != 0 || latest_checksum != checksum) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("latest model identity query failed: " + error);
    }
    if (client.FetchLatest(
            "aiserver-0", manifest, error) ||
        error != "downloaded model checksum mismatch") {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("corrupted transfer was not rejected: " + error);
    }

    service.SetWrongShape(true);
    error.clear();
    if (client.FetchLatest(
            "aiserver-0", manifest, error) ||
        error != "model manifest shape does not match AIServer") {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("shape mismatch was not rejected: " + error);
    }
    server->Shutdown();

    distribution.port = 1;
    distribution.rpc_timeout_ms = 100;
    ModelDistributorClient unavailable(distribution, model);
    error.clear();
    if (unavailable.FetchLatest(
            "aiserver-0", manifest, error) ||
        error.find("model manifest RPC failed") == std::string::npos) {
        fs::remove_all(root);
        return Fail("unreachable ModelDistributor was not reported");
    }

    const fs::path incoming_dir = root / "local-train" / "incoming";
    const fs::path active_dir = root / "local-train" / "active";
    fs::create_directories(incoming_dir);
    fs::create_directories(active_dir);
    {
        std::ofstream(active_dir / "model.onnx", std::ios::binary)
            << "old";
        std::ofstream(incoming_dir / "model_v000001.onnx", std::ios::binary)
            << "new";
    }
    manifest.model_path =
        (incoming_dir / "model_v000001.onnx").string();
    std::string previous_path;
    error.clear();
    if (!client.Promote(manifest, previous_path, error) ||
        !fs::is_regular_file(manifest.model_path) ||
        !fs::is_regular_file(previous_path)) {
        fs::remove_all(root);
        return Fail("model promotion did not preserve the previous model: " +
                    error);
    }

    fs::remove_all(root);
    return 0;
}
