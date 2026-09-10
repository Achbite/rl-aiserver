#include "maze/protocol/service.h"
#include "maze/observation/observation.h"
#include "task/sample/sample_sender.h"
#include "task/sample/training_transition_builder.h"
#include "task/sample/rollout_transition_builder.h"
#include "maze/environment/map.h"
#include "model_distributor_fixture.h"
#include <future>
#include <thread>

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
#include "proto/metrics/registry.pb.h"
#include "proto/metrics/transport.pb.h"

namespace {

void Require(bool condition, const std::string& message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
}

bool Near(float actual, double expected) {
    return std::fabs(static_cast<double>(actual) - expected) <= 1e-6;
}

class CapturingSamplePool final
    : public training::SamplePoolIngressService::Service {
public:
    grpc::Status GetStatus(
        grpc::ServerContext*,
        const training::SamplePoolStatusReq*,
        training::SamplePoolStatusRsp* response) override {
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

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::vector<training::ProcessedTransitionEnvelope> envelopes_;
};

MazeConfig MakeConfig(int sample_pool_port) {
    MazeConfig config;
    const int action_count = static_cast<int>(maze::MazeAction_MAX) + 1;
    config.server.run_mode = aiserver_mode::kTraining;
    config.policy.training_temperature = 1.0;
    config.policy.action_mask_mode = "disabled";
    config.model.expected_obs_dim = MazeObservation::kDimension;
    config.model.expected_action_dim = action_count;
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
    MazeConfig config = MakeConfig(1);
    config.server.run_mode = aiserver_mode::kEvaluation;
    config.environment.agent_count = 1;
    config.observation.ray_max_range = 2;
    config.model.evaluation_model_path = fixture_path;
    config.policy.action_mask_mode = "required";
    config.task.fixed_map_id = map.map_id();
    config.task.episode_max_steps = 10;
    MazeTaskService service(config);
    Require(service.Start(),
            "start the public evaluation inference service");

    maze::OpenSessionReq open_request;
    open_request.mutable_client()->set_component("maze-client");
    open_request.mutable_client()->set_instance_id("client-fixed");
    open_request.mutable_client()->set_lifecycle_epoch(1);
    open_request.set_environment_instance_id("environment-fixed");
    open_request.set_request_id("open-fixed");
    maze::OpenSessionRsp open_response;
    service.OpenSession(nullptr, &open_request, &open_response);
    Require(open_response.reply().result() ==
                rl::session::v1::COMMAND_RESULT_APPLIED &&
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
                rl::session::v1::COMMAND_RESULT_APPLIED,
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
                rl::session::v1::COMMAND_RESULT_APPLIED &&
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

    Require(response.reply().result() == rl::session::v1::COMMAND_RESULT_APPLIED &&
                response.reply().applied_sequence() == 3 &&
                response.action_batch().actions_size() == 1 &&
                response.action_batch().actions(0).agent_id() == 0 &&
                response.action_batch().actions(0).action_id() ==
                    maze::MAZE_ACTION_NOOP,
            "production Update applies the Client-owned action mask");
    Require(response.reply().phase() ==
                rl::session::v1::SESSION_PHASE_EPISODE_RUNNING,
            "the model-selected action commits through the public Update");
}

RawRolloutTransition MakeRawTransition(
    uint64_t action_step,
    float behavior_value) {
    RawRolloutTransition transition;
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
    std::vector<RawRolloutTransition> segment{
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
    envelope.mutable_behavior_model()->CopyFrom(behavior_model);
    for (const auto& item : processed) {
        envelope.add_samples()->CopyFrom(item);
    }
    return envelope;
}

void TestGaeAndSampleDelivery(const std::string& fixture_path) {
    TestModelOutputActionResponse(fixture_path);
    CapturingSamplePool sample_pool;
    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(
        "127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&sample_pool);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "start the in-process SamplePool test sink");

    MazeConfig config = MakeConfig(port);
    auto terminal = BuildSegment(
        "segment-terminal", 0.0,
        -0.18515, -0.3, 0.01485, 0.0);

    auto tmax = BuildSegment(
        "segment-tmax", 0.4,
        0.187288, 0.096, 0.387288, 0.396);

    SampleDistributor distributor(config.sample_distributor);
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


void TestRegisteredEpisodeMetrics() {
    MetricRegistry registry;
    MazeReward::RegisterMetrics(registry);
    RegisterMazeEpisodeMetrics(registry);
    registry.Register("task.maze.reward.changed_reward.per_transition", "Changed Reward", "reward", "agent_episode",
        training::METRIC_VALUE_TYPE_SUM_COUNT, training::METRIC_AGGREGATION_MEAN, "transition", "reward");
    registry.Register("task.maze.reward.changed_reward.per_episode", "Changed Reward / Agent Episode", "reward", "agent_episode",
        training::METRIC_VALUE_TYPE_SUM_COUNT, training::METRIC_AGGREGATION_MEAN, "agent_episode", "reward");
    AgentEpisodeResult first;
    first.agent_id = 0;
    first.episode_return = 3.0;
    first.transition_count = 4;
    first.success = true;
    first.termination_reason = maze::MAZE_TERMINATION_REASON_GOAL_REACHED;
    first.shortest_action_steps = 2;
    first.unique_cell_count = 3;
    first.blocked_move_count = 1;
    first.attempted_move_count = 4;
    first.behavior_model_lineage_id = "lineage";
    first.reward_component_sums["changed_reward"] = 3.0;
    auto second = first;
    second.agent_id = 1;
    second.episode_return = -1.0;
    second.transition_count = 2;
    second.success = false;
    second.reward_component_sums["changed_reward"] = -1.0;
    auto record = BuildMazeEpisodeMetrics(registry, "environment", "episode", {first, second});
    Require(std::none_of(record.points().begin(), record.points().end(), [](const auto& point) {
        return point.metric_id() == "task.maze.reward.total.per_transition";
    }), "Episode summary does not repeat per-transition rewards");
    RewardMetricWindow window;
    const auto started_at = std::chrono::steady_clock::time_point(std::chrono::milliseconds(10000));
    // A sparse component still counts the zero-reward transitions.
    for (double value : {3.0, 0.0, 0.0, 0.0, -1.0, 0.0})
        window.Observe(value, std::vector<std::pair<std::string, double>>{{"changed_reward", value}}, started_at);
    const auto ended_at = started_at + std::chrono::milliseconds(2500);
    window.AppendTo(record, registry, "task.maze.reward.", 1700000002500, ended_at);
    double sum = 0.0;
    double total_reward = 0.0;
    uint64_t total_transitions = 0;
    uint64_t transitions = 0;
    double episode_return_sum = 0.0;
    double component_episode_sum = 0.0;
    uint64_t agent_episodes = 0;
    uint64_t component_agent_episodes = 0;
    for (const auto& point : record.points()) {
        if (point.metric_id() == "task.maze.reward.total.per_transition") {
            Require(point.interval_start_unix_ms() == 1700000000000 &&
                    point.interval_end_unix_ms() == 1700000002500, "short tail retains its actual interval");
            total_reward += point.sum_count().sum();
            total_transitions += point.sum_count().count();
        }
        if (point.metric_id() == "task.maze.reward.changed_reward.per_transition") {
            sum += point.sum_count().sum();
            transitions += point.sum_count().count();
        }
        if (point.metric_id() == "task.maze.return") {
            episode_return_sum += point.sum_count().sum();
            agent_episodes += point.sum_count().count();
        }
        if (point.metric_id() == "task.maze.reward.changed_reward.per_episode") {
            component_episode_sum += point.sum_count().sum();
            component_agent_episodes += point.sum_count().count();
        }
    }
    Require(sum == 2.0 && transitions == 6, "task producer registers raw reward sums and actual transition denominators");
    Require(total_reward == 2.0 && total_transitions == 6,
            "Total Reward uses raw reward sums and actual transition counts, independently of Episode Return");
    Require(episode_return_sum == 2.0 && agent_episodes == 2 &&
            component_episode_sum == 2.0 && component_agent_episodes == 2,
            "completed agent episodes contribute one full return each, including failures and unequal lengths");
    // The closing wall clock moves behind the original opening wall time.
    // Raw rewards and elapsed duration must still reach the public journal.
    auto regressed = BuildMazeEpisodeMetrics(registry, "environment", "episode", {first, second});
    window.AppendTo(regressed, registry, "task.maze.reward.", 1699999999945, ended_at);
    for (const auto& point : regressed.points()) {
        if (point.metric_id() != "task.maze.reward.total.per_transition") continue;
        Require(point.interval_start_unix_ms() == 1699999997445 &&
                point.interval_end_unix_ms() == 1699999999945 &&
                point.sum_count().sum() == 2.0 && point.sum_count().count() == 6,
                "wall-clock regression preserves the elapsed interval and reward sum/count");
    }
    for (const auto& definition : record.definitions()) {
        if (definition.metric_id() == "task.maze.reward.total.per_transition") {
            Require(definition.display_name() == "Total Reward / Transition" && definition.denominator() == "transition",
                    "task registration owns the readable name and statistical denominator");
        }
        if (definition.metric_id() == "task.maze.return") {
            Require(definition.display_name() == "Mean Episode Return" &&
                    definition.category() == "reward" && definition.denominator() == "agent_episode",
                    "episode return is a registered reward field with an agent-episode denominator");
        }
    }
    common::ServiceInstanceIdentity producer;
    producer.set_component("aiserver");
    producer.set_instance_id("metric-producer");
    producer.set_lifecycle_epoch(1);
    MetricEventJournal journal(producer, 4096, 1024 * 1024, std::chrono::milliseconds(0));
    Require(journal.AppendFact(regressed.SerializeAsString(), 1699999999945).applied(),
            "generic metric journal accepts a registered task record");
    training::GetMetricBatchReq get;
    get.mutable_consumer()->set_component("learner");
    get.mutable_consumer()->set_instance_id("metric-reader");
    get.mutable_consumer()->set_lifecycle_epoch(1);
    *get.mutable_cursor()->mutable_source() = producer;
    get.set_max_events(10);
    get.set_max_bytes(1024 * 1024);
    training::GetMetricBatchRsp response;
    journal.Get(get, response);
    Require(response.has_batch() && response.batch().events_size() == 1 &&
                response.batch().events(0).fact_kind() == training::METRIC_FACT_KIND_REGISTERED_METRICS &&
                response.batch().events(0).fact_payload() == regressed.SerializeAsString(),
            "public journal Get preserves registered metric payload and current kind");
}

void TestModelFeedbackVisibility(const std::string& fixture_path) {
    model_fixture::TemporaryRoot root;
    const auto bytes = model_fixture::ReadFile(fixture_path);
    model_fixture::FixedModelDistributor model_sink(model_fixture::MakeManifest(bytes), bytes);
    CapturingSamplePool pool_sink;
    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&model_sink);
    builder.RegisterService(&pool_sink);
    auto server = builder.BuildAndStart();
    Require(server != nullptr && port > 0, "start adjacent model and sample sinks");
    auto config = MakeConfig(port);
    config.model.local_train_dir = (root.path() / "train").string();
    config.model_distribution.port = port;
    config.model_distribution.poll_interval_ms = 50;
    config.model_distribution.rpc_timeout_ms = 500;
    config.model.startup_timeout_ms = 2000;
    MazeTaskService service(config);
    Require(service.Start(), "start AIServer with an actual prepared ONNX model");
    model_sink.SetCandidate(1, "invalid ONNX candidate");
    training::AIServerStatusReq request;
    training::AIServerStatusRsp status;
    const auto failed_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    do {
        service.GetAIServerStatus(nullptr, &request, &status);
        if (!status.model_feedback().last_error().empty()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < failed_deadline);
    Require(status.ready() && status.loaded_model().model_step() == 0 &&
                status.model_feedback().candidate_model().model_step() == 1 &&
                status.model_feedback().stage() == "prepare" &&
                !status.model_feedback().last_error().empty(),
            "active model readiness and failed candidate preparation are independently observable");
    model_sink.SetCandidate(2, bytes);
    const auto recovered_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    do {
        service.GetAIServerStatus(nullptr, &request, &status);
        if (status.loaded_model().model_step() == 2) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < recovered_deadline);
    Require(status.loaded_model().model_step() == 2 && status.model_feedback().last_error().empty(),
            "a later successfully prepared publication clears the observed candidate failure");
    // Settle the real source-final ACK while shutdown waits for its consumer.
    auto shutdown = std::async(std::launch::async, [&] { return service.BeginShutdown(); });
    training::GetMetricBatchReq get;
    get.mutable_consumer()->set_component("learner");
    get.mutable_consumer()->set_instance_id("metric-test-consumer");
    get.mutable_consumer()->set_lifecycle_epoch(1);
    *get.mutable_cursor()->mutable_source() = service.MetricSourceIdentity();
    get.set_max_events(10);
    get.set_max_bytes(1024 * 1024);
    get.set_wait_timeout_ms(100);
    while (shutdown.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        training::GetMetricBatchRsp response;
        service.Metrics().GetMetricBatch(nullptr, &get, &response);
        if (response.has_batch()) {
            training::AckMetricBatchReq ack;
            *ack.mutable_consumer() = get.consumer();
            *ack.mutable_cursor()->mutable_source() = response.producer();
            ack.mutable_cursor()->set_acknowledged_batch_sequence(response.batch().batch_sequence());
            ack.mutable_cursor()->set_acknowledged_event_sequence(response.batch().final_event_sequence());
            training::AckMetricBatchRsp ack_response;
            service.Metrics().AckMetricBatch(nullptr, &ack, &ack_response);
            *get.mutable_cursor() = ack_response.committed_cursor();
        }
    }
    Require(shutdown.get(), "shutdown drains and settles actual final metric ACK");
    server->Shutdown();
    server->Wait();
}

}  // namespace

int main(int argc, char** argv) {
    Require(argc == 2,
            "usage: gae_sample_delivery_development_test MODEL");
    TestGaeAndSampleDelivery(argv[1]);
    TestRegisteredEpisodeMetrics();
    TestModelFeedbackVisibility(argv[1]);
    std::cout << "aiserver_gae_sample_delivery_data_path: PASS"
              << std::endl;
    return 0;
}
