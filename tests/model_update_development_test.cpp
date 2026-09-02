#include "ai/onnx_inferencer.h"
#include "ai/maze_observation.h"
#include "model/model_distributor_client.h"
#include "model/model_manifest.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <grpcpp/grpcpp.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    Require(input.is_open(), "open fixed ONNX fixture");
    std::ostringstream output;
    output << input.rdbuf();
    Require(!input.bad(), "read fixed ONNX fixture");
    return output.str();
}

AIServerConfig MakeConfig(const std::filesystem::path& root, int port) {
    AIServerConfig config;
    const int action_count = static_cast<int>(maze::MazeAction_MAX) + 1;
    config.policy.training_temperature = 1.0;
    config.policy.action_mask_mode = "disabled";
    config.model.expected_obs_dim = MazeObservation::kDimension;
    config.model.expected_action_dim = action_count;
    config.model.local_train_dir = (root / "train").string();
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    config.model_distribution.rpc_timeout_ms = 1000;
    return config;
}

training::ModelArtifactManifest MakeManifest(
    const std::string& model_bytes) {
    training::ModelArtifactManifest manifest;
    auto* identity = manifest.mutable_identity();
    identity->set_model_lineage_id("lineage-fixed");
    identity->set_model_step(0);
    manifest.set_size_bytes(static_cast<int64_t>(model_bytes.size()));
    manifest.set_trained_samples(0);
    manifest.set_published_at_unix_ms(1700000000000);
    return manifest;
}

class FixedModelDistributor final
    : public training::ModelDistributorService::Service {
public:
    FixedModelDistributor(training::ModelArtifactManifest manifest,
                          std::string model_bytes)
        : manifest_(std::move(manifest)),
          model_bytes_(std::move(model_bytes)) {}

    grpc::Status GetModelDistributorStatus(
        grpc::ServerContext*,
        const training::ModelDistributorStatusReq*,
        training::ModelDistributorStatusRsp* response) override {
        response->set_ready(true);
        FillAuthority(response->mutable_distributor());
        *response->mutable_latest_model() = manifest_.identity();
        response->set_available_floor_model_step(0);
        response->set_latest_available_model_step(0);
        return grpc::Status::OK;
    }

    grpc::Status GetModelManifest(
        grpc::ServerContext*,
        const training::GetModelManifestReq* request,
        training::GetModelManifestRsp* response) override {
        if (request->requested_model().model_lineage_id() !=
                manifest_.identity().model_lineage_id() ||
            !request->requested_model().has_model_step() ||
            request->requested_model().model_step() != 0) {
            response->set_result(training::MODEL_LOOKUP_RESULT_NOT_FOUND);
            response->set_message("fixed model not found");
            return grpc::Status::OK;
        }
        response->set_result(training::MODEL_LOOKUP_RESULT_FOUND);
        *response->mutable_manifest() = manifest_;
        FillAuthority(response->mutable_distributor());
        response->set_available_floor_model_step(0);
        response->set_latest_available_model_step(0);
        return grpc::Status::OK;
    }

    grpc::Status DownloadModel(
        grpc::ServerContext*,
        const training::DownloadModelReq* request,
        grpc::ServerWriter<training::ModelChunk>* writer) override {
        if (request->requested_model().SerializeAsString() !=
            manifest_.identity().SerializeAsString()) {
            return grpc::Status(
                grpc::StatusCode::NOT_FOUND, "fixed model not found");
        }
        training::ModelChunk chunk;
        *chunk.mutable_model() = manifest_.identity();
        chunk.set_offset(0);
        chunk.set_data(model_bytes_);
        writer->Write(chunk);
        return grpc::Status::OK;
    }

    grpc::Status AckModel(grpc::ServerContext*,
                          const training::AckModelReq* request,
                          training::AckModelRsp* response) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ack_ = *request;
        response->set_result(training::MODEL_ACK_RESULT_APPLIED);
        FillAuthority(response->mutable_distributor());
        return grpc::Status::OK;
    }

    training::AckModelReq ack() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ack_;
    }

private:
    static void FillAuthority(common::ServiceInstanceIdentity* identity) {
        identity->set_component("model-distributor");
        identity->set_instance_id("model-distributor-fixed");
        identity->set_lifecycle_epoch(1);
    }

    training::ModelArtifactManifest manifest_;
    std::string model_bytes_;
    mutable std::mutex mutex_;
    training::AckModelReq ack_;
};

class TemporaryRoot {
public:
    TemporaryRoot()
        : path_(std::filesystem::temp_directory_path() /
                ("aiserver-model-update-" + std::to_string(::getpid()))) {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~TemporaryRoot() { std::filesystem::remove_all(path_); }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

void TestModelUpdate(const std::string& fixture_path) {
    TemporaryRoot root;
    const std::string model_bytes = ReadFile(fixture_path);
    AIServerConfig config = MakeConfig(root.path(), 0);
    const auto wire_manifest = MakeManifest(model_bytes);
    FixedModelDistributor distributor(wire_manifest, model_bytes);

    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(
        "127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&distributor);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "start local ModelDistributor test service");
    config.model_distribution.port = port;

    ModelDistributorClient client(config, "aiserver-fixed", 1);
    std::string error;
    ModelDistributorClient::AvailableRange range;
    Require(client.GetAvailableRange("aiserver-fixed", range, error) &&
                range.floor_model_step == 0 &&
                range.latest_model_step == 0,
            "discover the fixed model identity: " + error);
    ModelManifest downloaded;
    Require(client.FetchStep("aiserver-fixed", 0, downloaded, error) &&
                downloaded.wire.identity().SerializeAsString() ==
                    wire_manifest.identity().SerializeAsString() &&
                ReadFile(downloaded.model_path) == model_bytes,
            "download and verify the fixed model: " + error);

    OnnxInferencer inferencer;
    OnnxInferencer::PreparedModel prepared;
    Require(inferencer.PrepareModel(
                downloaded.model_path, config.model.expected_obs_dim,
                config.model.expected_action_dim, prepared, &error) &&
                prepared.valid(),
            "Prepare the downloaded ONNX model: " + error);
    Require(client.PublishPrepared(downloaded, error),
            "publish the prepared model into the private cache: " + error);
    prepared.model_path = downloaded.model_path;
    Require(client.Ack(downloaded, "aiserver-fixed",
                       training::MODEL_LOAD_STATUS_LOADED, "loaded", error),
            "ACK the exact prepared model: " + error);
    inferencer.ActivatePreparedModel(std::move(prepared));

    std::vector<float> logits;
    float value = 0.0f;
    Require(inferencer.Infer(
                std::vector<float>(config.model.expected_obs_dim, 0.25f),
                config.model.expected_obs_dim, logits, value) &&
                logits.size() ==
                    static_cast<std::size_t>(config.model.expected_action_dim) &&
                std::isfinite(value),
            "the activated model produces finite logits and value");
    for (float logit : logits) {
        Require(std::isfinite(logit), "the activated model logit is finite");
    }

    const auto ack = distributor.ack();
    Require(ack.model().SerializeAsString() ==
                wire_manifest.identity().SerializeAsString() &&
                ack.load_status() == training::MODEL_LOAD_STATUS_LOADED,
            "the ACK belongs to the downloaded and activated model");

    server->Shutdown();
    server->Wait();
}

}  // namespace

int main(int argc, char** argv) {
    Require(argc == 2, "usage: model_update_development_test MODEL");
    TestModelUpdate(argv[1]);
    std::cout << "aiserver_model_update_data_path: PASS"
              << std::endl;
    return 0;
}
