#include "grpc/maze_service.h"
#include "ai/maze_observation.h"
#include "sample/sample_sender.h"
#include "sample/training_transition_builder.h"
#include "task/maze_map_contract.h"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <grpcpp/grpcpp.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
}

bool Near(float actual, double expected) {
    return std::fabs(static_cast<double>(actual) - expected) <= 1e-6;
}

void SetDigest(const std::string& hex, common::ContentDigest* digest) {
    digest->set_algorithm(common::DIGEST_ALGORITHM_SHA256);
    digest->set_hex(hex);
}

void FillContract(const ContractConfig& source,
                  common::ContractIdentity* destination) {
    destination->set_package_name(source.package_name);
    destination->set_package_version(source.package_version);
    destination->set_platform(source.platform);
}

class CapturingSamplePool final
    : public training::SamplePoolIngressService::Service {
public:
    explicit CapturingSamplePool(ContractConfig contract)
        : contract_(std::move(contract)) {}

    grpc::Status GetStatus(
        grpc::ServerContext*,
        const training::SamplePoolStatusReq*,
        training::SamplePoolStatusRsp* response) override {
        FillContract(contract_, response->mutable_contract());
        FillAuthority(response->mutable_sample_pool());
        response->set_ready(true);
        response->set_ingress_ready(true);
        response->set_pool_ready(true);
        response->set_backend_type(
            training::SAMPLE_BACKEND_TYPE_LOCAL_MEMORY);
        return grpc::Status::OK;
    }

    grpc::Status PushSamples(
        grpc::ServerContext*,
        const training::PushSamplesReq* request,
        training::PushSamplesRsp* response) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            envelopes_.push_back(request->envelope());
        }
        response->set_result(training::PUSH_RESULT_ACCEPTED);
        response->set_envelope_id(request->envelope().envelope_id());
        response->mutable_payload_digest()->CopyFrom(
            request->envelope().payload_digest());
        response->set_pressure_state(training::PRESSURE_STATE_NORMAL);
        FillAuthority(response->mutable_sample_pool());
        condition_.notify_all();
        return grpc::Status::OK;
    }

    bool WaitForEnvelopeCount(std::size_t expected) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(
            lock, std::chrono::seconds(2),
            [&] { return envelopes_.size() >= expected; });
    }

    std::vector<training::ProcessedTransitionEnvelope> envelopes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return envelopes_;
    }

private:
    static void FillAuthority(common::ServiceInstanceIdentity* identity) {
        identity->set_component("sample-pool");
        identity->set_instance_id("sample-pool-fixed");
        identity->set_lifecycle_epoch(1);
    }

    ContractConfig contract_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::vector<training::ProcessedTransitionEnvelope> envelopes_;
};

AIServerConfig MakeConfig(int sample_pool_port) {
    AIServerConfig config;
    const int action_count = static_cast<int>(maze::MazeAction_MAX) + 1;
    config.server.run_mode = aiserver_mode::kTraining;
    config.training_contract.training_contract_id = "maze.training";
    config.training_contract.observation_schema = {
        "maze.observation", 1, {"sha256", std::string(64, '4')}};
    config.training_contract.action_schema = {
        "maze.action", 1, {"sha256", std::string(64, '5')}};
    config.training_contract.reward_schema = {
        "maze.reward", 1, {"sha256", std::string(64, '6')}};
    config.training_contract.model_architecture_id =
        "actor-critic.independent-mlp";
    config.training_contract.canonical_digest.hex = std::string(64, '7');
    config.training_contract.observation_dimension =
        MazeObservation::kDimension;
    config.training_contract.action_count = action_count;
    config.training_contract.hidden_dimension =
        config.training_contract.observation_dimension;
    config.training_contract.tensor_dtype = "float32";
    config.training_contract.gae_formula_id = "gae.backward";
    config.training_contract.terminal_bootstrap_semantics_id =
        "maze.timeout-keep-and-cut-bootstrap";
    config.training_contract.value_target_formula_id =
        "advantage-plus-behavior-value";
    config.training_contract.value_head_abi_id = "scalar-value.float32";
    config.training_contract.numeric_dtype = "float32";
    config.training_contract.finite_rule_id = "reject-nonfinite";
    config.training_contract.model_pin_semantics_id =
        "per-agent-segment-pin";
    config.training_contract.action_mask_mode = "disabled";
    config.policy.training_temperature = 1.0;
    config.model.expected_obs_dim =
        config.training_contract.observation_dimension;
    config.model.expected_action_dim =
        config.training_contract.action_count;
    config.model.observation_schema_id = "maze.observation";
    config.model.action_schema_id = "maze.action";
    config.model.model_architecture_id =
        config.training_contract.model_architecture_id;
    config.model.tensor_dtype = "float32";
    config.sample_distributor.host = "127.0.0.1";
    config.sample_distributor.port = sample_pool_port;
    config.sample_distributor.envelope_max_transitions = 128;
    config.sample_distributor.envelope_max_bytes = 1024 * 1024;
    config.sample_distributor.health_timeout_ms = 500;
    config.sample_distributor.rpc_timeout_ms = 1000;
    config.sample_distributor.max_attempts = 1;
    config.sample_distributor.enqueue_timeout_ms = 100;
    config.sample_distributor.drain_timeout_ms = 2000;
    config.sample_distributor.status_poll_interval_ms = 100;
    config.sample_distributor.recovery_timeout_ms = 30000;
    config.sample_distributor.outbound_max_envelopes = 8;
    config.sample_distributor.outbound_max_estimated_bytes = 1024 * 1024;
    config.sample_distributor.aiserver_id = "aiserver-fixed";
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = 1;
    return config;
}

void TestModelOutputActionResponse(const std::string& fixture_path) {
    maze::MapDescriptor map;
    map.set_map_id("map-fixed");
    map.set_grid_columns(3);
    map.set_grid_rows(3);
    map.set_grid_size_microunits(1'000'000);
    map.set_start_grid_x(0);
    map.set_start_grid_y(0);
    map.set_goal_grid_x(2);
    map.set_goal_grid_y(2);
    map.set_blocked_bitmap(std::string(9, '\0'));
    AIServerConfig config = MakeConfig(1);
    config.server.run_mode = aiserver_mode::kEvaluation;
    config.environment.agent_count = 1;
    config.observation.ray_max_range = 2;
    config.model.evaluation_model_path = fixture_path;
    config.training_contract.action_mask_mode = "required";
    config.task.fixed_map_id = map.map_id();
    config.task.episode_max_steps = 10;
    MazeServiceImpl service(config);
    Require(service.Start(),
            "start the public evaluation inference service");

    maze::OpenSessionReq open_request;
    open_request.mutable_client()->set_component("maze-client");
    open_request.mutable_client()->set_instance_id("client-fixed");
    open_request.mutable_client()->set_lifecycle_epoch(1);
    open_request.set_environment_instance_id("environment-fixed");
    open_request.set_request_id("open-fixed");
    open_request.mutable_task_protocol()->set_protocol_id(
        config.task.task_protocol_id);
    open_request.mutable_task_protocol()->set_protocol_version(
        config.task.task_protocol_version);
    maze::OpenSessionRsp open_response;
    service.OpenSession(nullptr, &open_request, &open_response);
    Require(open_response.reply().result() ==
                maze::COMMAND_RESULT_APPLIED &&
                !open_response.session_id().empty(),
            "open the public evaluation Session");

    maze::InitReq init_request;
    init_request.mutable_command()->set_session_id(
        open_response.session_id());
    init_request.mutable_command()->set_session_epoch(
        open_response.session_epoch());
    init_request.mutable_command()->set_sequence(1);
    init_request.mutable_map()->CopyFrom(map);
    maze::InitRsp init_response;
    service.Init(nullptr, &init_request, &init_response);
    Require(init_response.reply().result() ==
                maze::COMMAND_RESULT_APPLIED,
            "initialize the public Session with the fixed map");

    maze::BeginEpisodeReq begin_request;
    begin_request.mutable_command()->set_session_id(
        open_response.session_id());
    begin_request.mutable_command()->set_session_epoch(
        open_response.session_epoch());
    begin_request.mutable_command()->set_sequence(2);
    maze::BeginEpisodeRsp begin_response;
    service.BeginEpisode(nullptr, &begin_request, &begin_response);
    Require(begin_response.reply().result() ==
                maze::COMMAND_RESULT_APPLIED &&
                begin_response.has_assignment() &&
                begin_response.assignment().mode() ==
                    maze::EPISODE_MODE_EVALUATION,
            "begin the public evaluation Episode");

    maze::UpdateReq request;
    request.mutable_command()->set_session_id(open_response.session_id());
    request.mutable_command()->set_session_epoch(
        open_response.session_epoch());
    request.mutable_command()->set_sequence(3);
    request.mutable_command()->set_episode_id(
        begin_response.assignment().episode_id());
    request.set_frame_id(0);
    auto* state = request.add_agents();
    state->set_agent_id(0);
    state->mutable_position()->set_x(0.0f);
    state->mutable_position()->set_y(0.0f);
    state->set_is_done(false);
    state->set_termination_reason(
        maze::MAZE_TERMINATION_REASON_ACTIVE);
    for (int action_id = maze::MazeAction_MIN;
         action_id <= maze::MazeAction_MAX; ++action_id) {
        state->add_action_mask(action_id == maze::MAZE_ACTION_NOOP);
    }

    maze::UpdateRsp response;
    service.Update(nullptr, &request, &response);

    Require(response.reply().result() == maze::COMMAND_RESULT_APPLIED &&
                response.reply().applied_sequence() == 3 &&
                response.action_batch().actions_size() == 1 &&
                response.action_batch().actions(0).agent_id() == 0 &&
                response.action_batch().actions(0).action_id() ==
                    maze::MAZE_ACTION_NOOP,
            "production Update applies the Client-owned action mask");
    Require(response.reply().phase() ==
                maze::SESSION_PHASE_EPISODE_RUNNING,
            "the model-selected action commits through the public Update");
}

SessionManager::RawRolloutTransition MakeRawTransition(
    uint64_t action_step,
    float behavior_value) {
    SessionManager::RawRolloutTransition transition;
    transition.observation.assign(
        MazeObservation::kDimension, static_cast<float>(action_step));
    transition.next_observation.assign(
        MazeObservation::kDimension,
        static_cast<float>(action_step + 1));
    const auto action_count =
        static_cast<uint64_t>(maze::MazeAction_MAX) + 1;
    transition.action = static_cast<int>(action_step % action_count);
    transition.reward = 0.0f;
    transition.behavior_log_probability = -0.5f;
    transition.behavior_value = behavior_value;
    transition.action_step = action_step;
    transition.created_at_unix_ms = 1700000000000 + action_step;
    return transition;
}

training::ProcessedTransitionEnvelope BuildSegment(
    const std::string& segment_id,
    double final_next_value,
    double expected_advantage_0,
    double expected_advantage_1,
    double expected_target_0,
    double expected_target_1) {
    std::vector<SessionManager::RawRolloutTransition> segment{
        MakeRawTransition(0, 0.2f),
        MakeRawTransition(1, 0.3f),
    };
    std::vector<float> advantages;
    std::vector<float> value_targets;
    std::string error;
    Require(EstimateRolloutSegment(
                segment, 0.99, 0.95, final_next_value,
                advantages, value_targets, error) &&
                advantages.size() == 2 && value_targets.size() == 2 &&
                Near(advantages[0], expected_advantage_0) &&
                Near(advantages[1], expected_advantage_1) &&
                Near(value_targets[0], expected_target_0) &&
                Near(value_targets[1], expected_target_1),
            "production GAE and value targets match the deterministic inputs: " +
                error);

    training::ModelIdentity behavior_model;
    behavior_model.set_model_lineage_id("lineage-test");
    behavior_model.set_model_step(0);
    SetDigest(std::string(64, '9'),
              behavior_model.mutable_artifact_digest());
    SetDigest(std::string(64, 'a'),
              behavior_model.mutable_manifest_digest());
    std::vector<training::ProcessedTransition> processed;
    Require(ProjectProcessedSegment(
                segment, advantages, value_targets, segment_id,
                behavior_model, MazeObservation::kDimension,
                static_cast<int>(maze::MazeAction_MAX) + 1,
                "disabled", processed, error) &&
                processed.size() == 2,
            "project the GAE result into the production training payload: " +
                error);

    training::ProcessedTransitionEnvelope envelope;
    envelope.set_envelope_id(segment_id + "/envelope-0");
    envelope.mutable_producer()->set_component("rl-aiserver");
    envelope.mutable_producer()->set_instance_id("aiserver-test");
    envelope.mutable_producer()->set_lifecycle_epoch(1);
    SetDigest(std::string(64, '7'),
              envelope.mutable_training_contract_digest());
    envelope.mutable_behavior_model()->CopyFrom(behavior_model);
    for (const auto& item : processed) {
        envelope.add_samples()->CopyFrom(item);
    }
    SetDigest(std::string(64, 'c'), envelope.mutable_payload_digest());
    return envelope;
}

void TestGaeAndSampleDelivery(const std::string& fixture_path) {
    TestModelOutputActionResponse(fixture_path);
    AIServerConfig seed = MakeConfig(1);
    CapturingSamplePool sample_pool(seed.contract);
    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(
        "127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&sample_pool);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "start the in-process SamplePool test sink");

    AIServerConfig config = MakeConfig(port);
    auto terminal = BuildSegment(
        "segment-terminal", 0.0,
        -0.18515, -0.3, 0.01485, 0.0);

    auto tmax = BuildSegment(
        "segment-tmax", 0.4,
        0.187288, 0.096, 0.387288, 0.396);

    SampleDistributor distributor(config);
    Require(distributor.Start(),
            "SampleDistributor connects to the in-process test sink");
    Require(distributor.Enqueue(terminal) && distributor.Enqueue(tmax),
            "both fixed envelopes enter the production SampleDistributor");
    Require(sample_pool.WaitForEnvelopeCount(2),
            "test sink receives both processed-transition envelopes");
    Require(distributor.StopAndDrain(),
            "SampleDistributor drains the fixed envelopes");

    const auto captured = sample_pool.envelopes();
    Require(captured.size() == 2 &&
                captured[0].SerializeAsString() == terminal.SerializeAsString() &&
                captured[1].SerializeAsString() == tmax.SerializeAsString() &&
                captured[0].samples_size() == 2 &&
                captured[1].samples_size() == 2,
            "test sink captures the exact two-transition envelopes");

    server->Shutdown();
    server->Wait();
}

}  // namespace

int main(int argc, char** argv) {
    Require(argc == 2,
            "usage: gae_sample_delivery_development_test MODEL");
    TestGaeAndSampleDelivery(argv[1]);
    std::cout << "aiserver_gae_sample_delivery_development_contract: PASS"
              << std::endl;
    return 0;
}
