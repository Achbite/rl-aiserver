#include "grpc/maze_service.h"
#include "sample/training_transition_builder.h"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <grpcpp/grpcpp.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct MazeServiceUpdateTestAccess {
    static bool PrepareSegment(
        MazeServiceImpl& service,
        SessionManager::Session& session,
        int agent_id,
        training::SegmentCloseReason reason,
        float bootstrap_value,
        bool bootstrap_applied,
        std::vector<training::ProcessedTransitionEnvelope>& envelopes,
        std::string& error) {
        int64_t produced_transitions = 0;
        int64_t produced_envelopes = 0;
        std::unordered_map<ModelStep, int64_t> produced_by_model;
        std::unordered_map<training::SegmentCloseReason, int64_t>
            close_counts;
        return service.PrepareAgentSegmentClose(
            session, agent_id, reason, bootstrap_value, bootstrap_applied,
            produced_transitions, produced_envelopes, produced_by_model,
            envelopes, close_counts, error);
    }

    static bool PrepareFixtureModel(
        MazeServiceImpl& service,
        const std::string& model_path,
        OnnxInferencer::PreparedModel& prepared,
        std::string& error) {
        return service.onnx_inferencer_.PrepareModel(
            model_path, service.config_.model.expected_obs_dim,
            service.config_.model.expected_action_dim, prepared, &error);
    }

    static SampleDistributor& Distributor(MazeServiceImpl& service) {
        return service.sample_distributor_;
    }
};

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
    SetDigest(source.source_digest.hex, destination->mutable_source_digest());
    SetDigest(source.artifact_digest.hex,
              destination->mutable_artifact_digest());
    destination->set_platform(source.platform);
    destination->set_generator_identity(source.generator_identity);
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
        response->set_ret_code(0);
        response->set_result(training::PUSH_RESULT_ACCEPTED);
        response->set_envelope_id(request->envelope().envelope_id());
        response->set_accepted_transitions(
            request->envelope().transitions_size());
        response->set_accepted_unique_transitions(
            request->envelope().transitions_size());
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
    config.server.run_mode = aiserver_mode::kTraining;
    config.contract.source_digest.hex = std::string(64, '1');
    config.contract.artifact_digest.hex = std::string(64, '2');
    config.contract.generator_identity = std::string(64, '3');
    config.training_semantics.observation_schema = {
        "maze.observation.v3", 1, {"sha256", std::string(64, '4')}};
    config.training_semantics.action_schema = {
        "maze.action.v1", 1, {"sha256", std::string(64, '5')}};
    config.training_semantics.reward_schema = {
        "maze.reward.v4", 1, {"sha256", std::string(64, '6')}};
    config.training_semantics.semantics_digest.hex = std::string(64, '7');
    config.policy.policy_spec_digest.hex = std::string(64, '8');
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

ModelManifest MakePinnedModel(
    const OnnxInferencer::PreparedModel& prepared) {
    ModelManifest manifest;
    manifest.model_lineage_id = "lineage-fixed";
    manifest.model_step = 0;
    manifest.model_path = prepared.model_path;
    manifest.wire.mutable_identity()->set_model_lineage_id(
        manifest.model_lineage_id);
    manifest.wire.mutable_identity()->set_model_step(0);
    SetDigest(std::string(64, '9'),
              manifest.wire.mutable_identity()->mutable_artifact_digest());
    SetDigest(std::string(64, 'a'),
              manifest.wire.mutable_identity()->mutable_manifest_digest());
    auto* profile = manifest.wire.mutable_rollout_estimator_profile();
    profile->set_gamma(0.99);
    profile->set_gae_lambda(0.95);
    profile->set_tmax(128);
    SetDigest(std::string(64, 'b'), profile->mutable_profile_digest());
    return manifest;
}

SessionManager::RawRolloutTransition MakeRawTransition(
    uint64_t action_step,
    float behavior_value) {
    SessionManager::RawRolloutTransition transition;
    transition.observation.assign(17, static_cast<float>(action_step));
    transition.next_observation.assign(
        17, static_cast<float>(action_step + 1));
    transition.action = static_cast<int>(action_step % 9);
    transition.reward = 0.0f;
    transition.behavior_log_probability = -0.5f;
    transition.behavior_value = behavior_value;
    transition.action_step = action_step;
    transition.created_at_unix_ms = 1700000000000 + action_step;
    return transition;
}

training::ProcessedTransitionEnvelope BuildSegment(
    MazeServiceImpl& service,
    const OnnxInferencer::PreparedModel& prepared,
    const std::string& segment_id,
    training::SegmentCloseReason close_reason,
    float bootstrap_value,
    bool bootstrap_applied) {
    SessionManager::Session session;
    session.session_id = "session-fixed";
    session.environment_instance_id = "environment-fixed";
    session.current_episode_id = "episode-fixed";
    auto& agent = session.agents[1];
    agent.segment_open = true;
    agent.segment_id = segment_id;
    agent.pinned_model = MakePinnedModel(prepared);
    agent.pinned_prepared_model = prepared;
    agent.segment_transitions.push_back(MakeRawTransition(0, 0.2f));
    agent.segment_transitions.push_back(MakeRawTransition(1, 0.3f));

    std::vector<training::ProcessedTransitionEnvelope> envelopes;
    std::string error;
    Require(MazeServiceUpdateTestAccess::PrepareSegment(
                service, session, 1, close_reason, bootstrap_value,
                bootstrap_applied, envelopes, error) &&
                envelopes.size() == 1 &&
                envelopes.front().transitions_size() == 2,
            "build the fixed processed-transition envelope: " + error);
    return envelopes.front();
}

void CheckBoundary(
    const training::ProcessedTransitionEnvelope& envelope,
    training::SegmentCloseReason close_reason,
    bool terminal,
    bool bootstrap_applied,
    float bootstrap_value,
    double expected_advantage_0,
    double expected_advantage_1,
    double expected_target_0,
    double expected_target_1) {
    const auto& first = envelope.transitions(0);
    const auto& last = envelope.transitions(1);
    Require(Near(first.advantage(), expected_advantage_0) &&
                Near(last.advantage(), expected_advantage_1) &&
                Near(first.value_target(), expected_target_0) &&
                Near(last.value_target(), expected_target_1),
            "production GAE and value targets match the fixed inputs");
    Require(!first.segment_boundary() &&
                first.segment_close_reason() ==
                    training::SEGMENT_CLOSE_REASON_UNSPECIFIED &&
                !first.has_bootstrap_value() &&
                !first.bootstrap_applied(),
            "only the final transition carries close/bootstrap facts");
    Require(last.segment_boundary() &&
                last.segment_close_reason() == close_reason &&
                last.environment_terminal() == terminal &&
                last.has_bootstrap_value() &&
                last.bootstrap_applied() == bootstrap_applied &&
                Near(last.bootstrap_value(), bootstrap_value),
            "the final transition carries the explicit close/bootstrap facts");
}

void TestGaeAndSampleDelivery(const std::string& fixture_path) {
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
    MazeServiceImpl service(config);
    OnnxInferencer::PreparedModel prepared;
    std::string error;
    Require(MazeServiceUpdateTestAccess::PrepareFixtureModel(
                service, fixture_path, prepared, error) && prepared.valid(),
            "prepare the fixed pinned model: " + error);

    auto terminal = BuildSegment(
        service, prepared, "segment-terminal",
        training::SEGMENT_CLOSE_REASON_GOAL, 0.0f, false);
    CheckBoundary(terminal, training::SEGMENT_CLOSE_REASON_GOAL,
                  true, false, 0.0f,
                  -0.18515, -0.3, 0.01485, 0.0);

    auto tmax = BuildSegment(
        service, prepared, "segment-tmax",
        training::SEGMENT_CLOSE_REASON_TMAX, 0.4f, true);
    CheckBoundary(tmax, training::SEGMENT_CLOSE_REASON_TMAX,
                  false, true, 0.4f,
                  0.187288, 0.096, 0.387288, 0.396);

    auto& distributor = MazeServiceUpdateTestAccess::Distributor(service);
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
                captured[0].transitions_size() == 2 &&
                captured[1].transitions_size() == 2,
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
