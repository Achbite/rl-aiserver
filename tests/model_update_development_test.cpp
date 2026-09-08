#include "ai/onnx_inferencer.h"
#include "ai/maze_observation.h"
#include "model/model_distributor_client.h"
#include "model/model_manifest.h"
#include "model_distributor_fixture.h"

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

using namespace model_fixture;

void Require(bool condition, const std::string& message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
}

TrainingRuntimeConfig MakeConfig(const std::filesystem::path& root, int port) {
    TrainingRuntimeConfig config;
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

void TestModelUpdate(const std::string& fixture_path) {
    TemporaryRoot root;
    const std::string model_bytes = ReadFile(fixture_path);
    TrainingRuntimeConfig config = MakeConfig(root.path(), 0);
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

    ModelDistributorClient client(
        config.model_distribution, config.model, "aiserver-fixed", 1);
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
