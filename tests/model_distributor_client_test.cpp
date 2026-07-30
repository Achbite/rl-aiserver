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

    void SetWrongRun(bool enabled) {
        wrong_run_ = enabled;
    }

    grpc::Status GetModelManifest(
        grpc::ServerContext*,
        const maze::GetModelManifestReq* request,
        maze::GetModelManifestRsp* response) override {
        response->set_ret_code(0);
        auto* manifest = response->mutable_manifest();
        manifest->set_schema_version(1);
        manifest->set_contract_version("0.3.0");
        manifest->set_run_id(
            wrong_run_ ? "different-run" : request->run_id());
        manifest->set_model_version(0);
        manifest->set_artifact_uri("file:///models/model_v000000.onnx");
        manifest->set_model_file("model_v000000.onnx");
        manifest->set_size_bytes(4);
        manifest->set_sha256(checksum_);
        manifest->add_input_shape(1);
        manifest->add_input_shape(13);
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
        chunk.set_run_id(request->run_id());
        chunk.set_model_version(request->model_version());
        chunk.set_offset(0);
        chunk.set_data("evil");
        writer->Write(chunk);
        return grpc::Status::OK;
    }

private:
    std::string checksum_;
    bool wrong_run_ = false;
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
    distribution.contract_version = "0.3.0";
    ModelConfig model;
    model.p2p_dir = (root / "cache").string();
    ModelDistributorClient client(distribution, model);
    ModelManifest manifest;
    std::string error;
    int latest_version = -1;
    std::string latest_checksum;
    if (!client.GetLatestIdentity(
            "test-run", "aiserver-0",
            latest_version, latest_checksum, error) ||
        latest_version != 0 || latest_checksum != checksum) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("latest model identity query failed: " + error);
    }
    if (client.FetchLatest(
            "test-run", "aiserver-0", manifest, error) ||
        error != "downloaded model checksum mismatch") {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("corrupted transfer was not rejected: " + error);
    }

    service.SetWrongRun(true);
    error.clear();
    if (client.FetchLatest(
            "test-run", "aiserver-0", manifest, error) ||
        error != "model manifest run_id mismatch") {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("cross-run manifest was not rejected: " + error);
    }
    server->Shutdown();

    distribution.port = 1;
    distribution.rpc_timeout_ms = 100;
    ModelDistributorClient unavailable(distribution, model);
    error.clear();
    if (unavailable.FetchLatest(
            "test-run", "aiserver-0", manifest, error) ||
        error.find("model manifest RPC failed") == std::string::npos) {
        fs::remove_all(root);
        return Fail("unreachable ModelDistributor was not reported");
    }

    fs::remove_all(root);
    return 0;
}
