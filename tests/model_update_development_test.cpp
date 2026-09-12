#include "task/inference/onnx_inferencer.h"
#include "maze/protocol/service.h"
#include "task/runtime/training_transaction.h"
#include "task/session/session_manager.h"
#include "rl_sdk/task_client.h"
#include "task/model/model_distributor_client.h"
#include "task/model/model_manifest.h"
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

MazeConfig MakeConfig(const std::filesystem::path& root, int port) {
    MazeConfig config;
    const int action_count = static_cast<int>(maze::MazeAction_MAX) + 1;
    config.policy.training_temperature = 1.0;
    config.policy.action_mask_mode = "disabled";
    config.model.expected_obs_dim = 17;
    config.model.expected_action_dim = action_count;
    config.model.local_train_dir = (root / "train").string();
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    config.model_distribution.rpc_timeout_ms = 100;
    config.model_distribution.poll_interval_ms = 20;
    config.model.startup_timeout_ms = 700;
    config.server.run_mode = aiserver_mode::kTraining;
    config.sample_distributor.host = "127.0.0.1";
    config.sample_distributor.port = port;
    config.sample_distributor.health_timeout_ms = 500;
    config.sample_distributor.rpc_timeout_ms = 500;
    config.sample_distributor.drain_timeout_ms = 2000;
    config.environment.agent_count = 1;
    config.observation.ray_max_range = 2;
    config.task.fixed_map_id = "test-map";
    config.task.episode_max_steps = 10;
    return config;
}

void TestModelUpdate(const std::string& fixture_path) {
    TemporaryRoot root;
    const std::string model_bytes = ReadFile(fixture_path);
    auto config = MakeConfig(root.path(), 0);
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
                std::fabs(value - .25f) < 1e-6 &&
                std::fabs(logits[1] - 1.0f) < 1e-6 && std::fabs(logits[3] - .5f) < 1e-6,
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

using Client = rl_sdk::TaskClient<maze::MazeTaskServiceProtocol>;

void BeginIdleEpisode(Client& client, int port) {
    Require(client.Connect("127.0.0.1:" + std::to_string(port)), "connect SDK for idle lifecycle");
    maze::OpenSessionRsp open;
    Require(client.OpenSession(open) == rl_sdk::CommandOutcome::Applied, client.error());
    maze::InitReq init;
    auto* map = init.mutable_map();
    map->set_map_id("test-map");
    map->set_grid_columns(3); map->set_grid_rows(3);
    map->set_grid_size_microunits(1000000);
    map->set_start_grid_x(0); map->set_start_grid_y(0);
    map->set_goal_grid_x(2); map->set_goal_grid_y(2);
    map->set_blocked_bitmap(std::string(9, '\0'));
    maze::InitRsp initialized;
    maze::BeginEpisodeRsp begin;
    Require(client.Init(init, initialized) == rl_sdk::CommandOutcome::Applied &&
            client.BeginEpisode(begin) == rl_sdk::CommandOutcome::Applied, client.error());
}

template<class Predicate>
training::AIServerStatusRsp AwaitStatus(MazeTaskService& service, Predicate satisfied) {
    training::AIServerStatusReq request;
    training::AIServerStatusRsp response;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
    do {
        service.GetAIServerStatus(nullptr, &request, &response);
        if (satisfied(response)) return response;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);
    Require(false, "expected AIServer status not reached: " + response.DebugString());
    return response;
}

void TestCandidateAck(const std::string& fixture_path, bool recover) {
    TemporaryRoot root;
    const auto bytes = ReadFile(fixture_path);
    FixedModelDistributor models(MakeManifest(bytes), bytes);
    CapturingSamplePool pool;
    LocalServer adjacent({&models, &pool});
    auto config = MakeConfig(root.path(), adjacent.port);
    config.sample_distributor.recovery_timeout_ms = 1600;
    MazeTaskService service(config);
    Require(service.Start(), "initial model really becomes ready before candidate ACK test");
    LocalServer server({static_cast<maze::MazeTaskService::Service*>(&service)});
    Client client(model_fixture::ClientOptions());
    BeginIdleEpisode(client, server.port);
    const auto episode_started = std::chrono::steady_clock::now();
    models.SetUncertainAcks(-1);
    models.SetCandidate(1, bytes);
    // The first actual LOADED ACK is unknown; active model must remain step zero.
    const auto ack_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (models.ack().model().model_step() != 1 && std::chrono::steady_clock::now() < ack_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    Require(models.ack().model().model_step() == 1, "candidate reaches the real ACK operation");
    auto status = AwaitStatus(service, [](const auto& value) { return value.loaded_model().model_step() == 0; });
    Require(status.model_state() != training::MODEL_STATE_FAILED, "unknown ACK cannot activate the candidate");
    if (recover) {
        models.SetUncertainAcks(0);
        status = AwaitStatus(service, [](const auto& value) { return value.loaded_model().model_step() == 1; });
        Require(status.ready(), "candidate activates when ACK confirms within its budget");
        maze::AbortEpisodeReq abort;
        abort.set_reason(maze::MAZE_TERMINATION_REASON_CLIENT_ABORT);
        maze::AbortEpisodeRsp aborted;
        maze::CloseSessionRsp closed;
        Require(client.AbortEpisode(abort, aborted) == rl_sdk::CommandOutcome::Applied &&
                client.CloseSession(closed) == rl_sdk::CommandOutcome::Applied, client.error());
    } else {
        status = AwaitStatus(service, [](const auto& value) { return value.model_state() == training::MODEL_STATE_FAILED; });
        Require(!status.ready() && status.model_feedback().stage() == "ack_unconfirmed" &&
                status.model_feedback().last_error().find("ACK receipt unavailable") != std::string::npos &&
                status.loaded_model().model_step() == 0,
                "whole-operation expiry preserves the unknown ACK cause without candidate activation");
        const int calls = models.io_calls();
        models.SetUncertainAcks(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        Require(models.io_calls() == calls, "terminal model chain stops model I/O even when peer recovers");
        // A running episode cannot Close. After the existing idle deadline, a
        // successful public Close proves the watcher still performed cleanup.
        std::this_thread::sleep_until(episode_started + std::chrono::milliseconds(2200));
        maze::CloseSessionRsp closed;
        Require(client.CloseSession(closed) == rl_sdk::CommandOutcome::Applied,
                "expired Client episode can Close after model-chain failure: " + client.error());
        Require(models.io_calls() == calls, "Client timeout cleanup does not restart model I/O");
    }
    std::string ack_payload;
    for (const auto& ack : models.acks()) {
        if (ack.model().model_step() != 1) continue;
        Require(ack.load_status() == training::MODEL_LOAD_STATUS_LOADED, "unknown LOADED is never overwritten by FAILED");
        if (ack_payload.empty()) ack_payload = ack.SerializeAsString();
        Require(ack.SerializeAsString() == ack_payload, "retry preserves exact model, authority, source and payload");
    }
    Require(!ack_payload.empty(), "candidate ACK was actually sent");
    const auto times = models.candidate_ack_times();
    Require(!times.empty(), "candidate RPC deadlines were observed at the neighboring service");
    for (const auto& attempt : times) {
        // gRPC timeout encoding rounds sub-millisecond durations. This checks
        // the existing operation budget, not a response-time target.
        Require(attempt.second <= times.front().first +
                    std::chrono::milliseconds(config.model.startup_timeout_ms + 10),
                "each actual RPC deadline stays within the candidate operation budget");
    }
    const bool clean = FinishMetrics([&] { return service.BeginShutdown(); }, service.Metrics(), service.MetricSourceIdentity());
    Require(clean == recover, "shutdown preserves an existing model fault");
}

void TestInitialPrepareFailure(const std::string& fixture_path) {
    TemporaryRoot root;
    const auto bytes = ReadFile(fixture_path);
    FixedModelDistributor models(MakeManifest(bytes), bytes);
    CapturingSamplePool pool;
    LocalServer adjacent({&models, &pool});
    auto config = MakeConfig(root.path(), adjacent.port);
    config.model.expected_obs_dim = 18;
    MazeTaskService service(config);
    Require(!service.Start(), "actual incompatible ONNX shape fails initial loading");
    const auto ack = models.ack();
    Require(ack.model().SerializeAsString() == MakeManifest(bytes).identity().SerializeAsString() &&
            ack.load_status() == training::MODEL_LOAD_STATUS_FAILED && !ack.message().empty() &&
            ack.message().find("ONNX") != std::string::npos && ack.message().find("shape") != std::string::npos && !ack.aiserver().instance_id().empty(),
            "actual initial FAILED ACK carries model zero, AIServer source and prepare cause");
    Require(!FinishMetrics([&] { return service.BeginShutdown(); }, service.Metrics(), service.MetricSourceIdentity()),
            "startup fault remains visible through shutdown");
}

void TestRuntimeInferenceError(const std::string& fixture_path) {
    using Sessions = SessionManager<TrainingSession<AgentTrainingState>>;
    TemporaryRoot root;
    const auto bytes = ReadFile(fixture_path);
    FixedModelDistributor models(MakeManifest(bytes), bytes);
    CapturingSamplePool pool;
    LocalServer adjacent({&models, &pool});
    TrainingRuntime<Sessions> runtime(MakeConfig(root.path(), adjacent.port), "test.reward.");
    Require(runtime.Start(), "runtime-error fixture loads as a valid model");
    Sessions::Session session;
    session.session_id = "inference-error";
    session.current_episode_id = "episode";
    session.current_episode_mode = PolicyMode::Training;
    session.behavior_policy_scope = BehaviorPolicyScope::TrainingAgentSegment;
    session.agents.emplace(0, AgentTrainingState{});
    std::string error;
    {
        std::lock_guard<std::mutex> lock(runtime.mutex_);
        AgentTaskInput input;
        input.agent_id = 0;
        input.observation.assign(17, 0);
        input.observation[0] = 2;  // Actual Gather node accesses past its one-element data.
        TrainingTransaction<Sessions> transaction(runtime, session);
        std::vector<ModelTaskAction> actions;
        Require(!transaction.PrepareFrame({input}, 0, true, actions, error) && actions.empty() &&
                error.find("ValueIndex") != std::string::npos,
                "actual ONNX exception context reaches the transaction error: " + error);
    }
    training::AIServerStatusReq request;
    training::AIServerStatusRsp status;
    runtime.GetAIServerStatus(nullptr, &request, &status);
    Require(status.last_error().find(error) != std::string::npos && !runtime.sample_distributor_.GetSnapshot().degraded &&
            runtime.sample_distributor_.GetSnapshot().last_error.empty(),
            "runtime reports original inference cause without inventing a SampleSender fault");
    Require(!FinishMetrics([&] { return runtime.BeginShutdown(); }, runtime.metric_service_, runtime.MetricSourceIdentity()),
            "inference failure remains visible through cleanup");
}

}  // namespace

int main(int argc, char** argv) {
    Require(argc == 3, "usage: model_update_development_test MODEL ERROR_MODEL");
    TestModelUpdate(argv[1]);
    TestInitialPrepareFailure(argv[1]);
    TestCandidateAck(argv[1], false);
    TestCandidateAck(argv[1], true);
    TestRuntimeInferenceError(argv[2]);
    std::cout << "aiserver_model_update_data_path: PASS"
              << std::endl;
    return 0;
}
