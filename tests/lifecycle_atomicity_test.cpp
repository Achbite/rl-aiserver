#include "grpc/maze_service.h"

#include "task/maze_map_contract.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

struct MazeServiceUpdateTestAccess {
    static void SetRuntimeReady(MazeServiceImpl& service, bool degraded) {
        service.state_.store(training::AISERVER_STATE_READY);
        service.model_state_.store(training::MODEL_STATE_READY);
        std::lock_guard<std::mutex> lock(service.sample_sender_.mutex_);
        service.sample_sender_.accepting_ = true;
        service.sample_sender_.ready_ = true;
        service.sample_sender_.degraded_ = degraded;
        service.sample_sender_.delivery_state_ =
            degraded
                ? SampleSender::DeliveryState::kTerminalFault
                : SampleSender::DeliveryState::kHealthy;
        service.sample_sender_.transient_retry_after_ms_ = 0;
    }

    static bool SenderDegraded(const MazeServiceImpl& service) {
        return service.sample_sender_.IsDegraded();
    }
};

struct MazeServiceLifecycleTestAccess {
    static SessionManager::Session* AddSession(MazeServiceImpl& service) {
        const std::string id = service.session_mgr_.CreateSession();
        return service.session_mgr_.GetSession(id);
    }

    static void SetModel(MazeServiceImpl& service,
                         int version,
                         int64_t train_updates,
                         int64_t trained_samples) {
        service.model_manifest_.model_version = version;
        service.model_manifest_.sha256 =
            std::string(64, static_cast<char>('a' + version % 6));
        service.model_manifest_.model_lineage_id = "lifecycle-atomicity";
        service.model_manifest_.manifest_digest = std::string(64, 'f');
        service.model_manifest_.train_updates = train_updates;
        service.model_manifest_.trained_samples = trained_samples;
        auto* identity = service.model_manifest_.wire.mutable_identity();
        identity->set_model_lineage_id("lifecycle-atomicity");
        identity->set_model_version(static_cast<uint64_t>(version));
        identity->mutable_artifact_digest()->set_algorithm(
            common::DIGEST_ALGORITHM_SHA256);
        identity->mutable_artifact_digest()->set_hex(
            service.model_manifest_.sha256);
        identity->mutable_manifest_digest()->set_algorithm(
            common::DIGEST_ALGORITHM_SHA256);
        identity->mutable_manifest_digest()->set_hex(
            service.model_manifest_.manifest_digest);
        service.model_manifest_.wire.set_train_updates(train_updates);
        service.model_manifest_.wire.set_trained_samples(trained_samples);
    }

    static void SetModel(MazeServiceImpl& service,
                         int version,
                         int64_t trained_samples) {
        SetModel(service, version, version, trained_samples);
    }

    static void InstallReceiptWriter(MazeServiceImpl& service,
                                     bool& fail,
                                     int& calls,
                                     std::string& published) {
        service.task_controller_receipt_writer_ =
            [&](const SingleMapTaskController& candidate,
                std::string& error) {
                ++calls;
                if (fail) {
                    error = "injected task receipt write failure";
                    return false;
                }
                published = candidate.ToJson();
                return true;
            };
    }

    static std::string ControllerJson(const MazeServiceImpl& service) {
        return service.task_controller_.ToJson();
    }

    static SingleMapTaskSnapshot ControllerSnapshot(
        const MazeServiceImpl& service) {
        return service.task_controller_.GetSnapshot();
    }

    static bool ClientInitialized(const MazeServiceImpl& service) {
        return service.client_initialized_;
    }

    static bool TaskStopRequested(const MazeServiceImpl& service) {
        return service.task_stop_requested_;
    }

    static void SetModelAckPending(MazeServiceImpl& service, bool pending) {
        service.model_ack_pending_ = pending;
    }

    static bool ModelAckPending(const MazeServiceImpl& service) {
        return service.model_ack_pending_;
    }

    static void SetClientActivity(SessionManager::Session& session,
                                  int64_t timestamp_unix_ms) {
        session.last_valid_client_activity_unix_ms = timestamp_unix_ms;
    }

    static int ActiveSessionCount(const MazeServiceImpl& service) {
        return service.session_mgr_.GetActiveSessionCount();
    }

    static std::string LastError(const MazeServiceImpl& service) {
        return service.last_error_;
    }

    static training::AIServerState ServiceState(
        const MazeServiceImpl& service) {
        return service.state_.load();
    }

    static uint64_t CompletedAgentCount(const MazeServiceImpl& service) {
        training::MetricSnapshot snapshot;
        common::ServiceInstanceIdentity source;
        source.set_component("rl-aiserver");
        source.set_instance_id("lifecycle-test");
        source.set_lifecycle_epoch(1);
        service.episode_metrics_.Fill(&snapshot, source, 1, 1);
        for (const auto& value : snapshot.values()) {
            if (value.field_id() ==
                "server.episode.learning_return.mean.v1") {
                return value.count();
            }
        }
        return 0;
    }

};

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

maze::MapDescriptor MapDescriptor() {
    maze::MapDescriptor map;
    map.set_map_id("lifecycle-map");
    map.set_format_version(4);
    map.set_grid_columns(3);
    map.set_grid_rows(3);
    map.set_grid_size_microunits(1000000);
    map.set_start_grid_x(0);
    map.set_start_grid_y(0);
    map.set_goal_grid_x(2);
    map.set_goal_grid_y(2);
    map.set_blocked_bitmap(std::string(9, '\0'));
    map.set_shortest_action_steps(2);
    map.set_action_rule_id("maze.action.9-way.no-corner-cut.v1");
    std::string error;
    map.mutable_canonical_digest()->set_algorithm(
        common::DIGEST_ALGORITHM_SHA256);
    map.mutable_canonical_digest()->set_hex(
        CanonicalMazeMapChecksum(map, error));
    Require(error.empty() && map.canonical_digest().hex().size() == 64,
            "fixture map digest is valid");
    return map;
}

AIServerConfig Config(const maze::MapDescriptor& map) {
    AIServerConfig config;
    config.server.run_mode = aiserver_mode::kTraining;
    config.task.agent_num = 2;
    config.task.fixed_map_id = map.map_id();
    config.task.fixed_map_checksum_sha256 =
        map.canonical_digest().hex();
    config.task.action_rule_id = map.action_rule_id();
    config.task.shortest_action_steps =
        static_cast<int>(map.shortest_action_steps());
    config.sample_output.fragment_samples = 1;
    config.metrics.episode_window = 8;
    return config;
}

void SetTaskIdentity(SessionManager::Session& session,
                     const AIServerConfig& config) {
    session.task.set_task_contract_id(config.task.task_contract_id);
    session.task.set_task_id(config.task.task_id);
    session.task.set_task_revision(config.task.task_revision);
    session.task.mutable_task_config_digest()->set_algorithm(
        common::DIGEST_ALGORITHM_SHA256);
    session.task.mutable_task_config_digest()->set_hex(
        config.task.task_config_digest.hex);
}

void FillCommand(const SessionManager::Session& session,
                 uint64_t sequence,
                 const std::string& key,
                 maze::LifecycleCommand* command) {
    command->mutable_task()->CopyFrom(session.task);
    command->set_session_id(session.session_id);
    command->set_episode_id(session.current_episode_id);
    command->set_evaluation_id(session.current_evaluation_id);
    command->set_lifecycle_epoch(session.lifecycle_epoch);
    command->set_command_sequence(sequence);
    command->set_idempotency_key(key);
    command->set_expected_task_state(session.task_state);
    command->set_expected_session_state(session.session_state);
    command->set_expected_episode_state(session.protocol_episode_state);
    command->set_expected_evaluation_state(session.evaluation_state);
}

void ConfigureInitSession(SessionManager::Session& session,
                          const AIServerConfig& config) {
    session.lifecycle_epoch = 7;
    SetTaskIdentity(session, config);
    session.map_id = config.task.fixed_map_id;
    session.opened = true;
    session.initialized = false;
    session.task_state = maze::TASK_STATE_INITIALIZING;
    session.session_state = maze::SESSION_STATE_OPENED;
    session.protocol_episode_state = maze::EPISODE_STATE_UNSPECIFIED;
    session.evaluation_state = maze::EVALUATION_STATE_INACTIVE;
}

void ConfigureTrainingEndSession(SessionManager::Session& session,
                                 const AIServerConfig& config) {
    session.lifecycle_epoch = 11;
    SetTaskIdentity(session, config);
    session.opened = true;
    session.initialized = true;
    session.task_state = maze::TASK_STATE_TRAINING;
    session.session_state = maze::SESSION_STATE_EPISODE_ACTIVE;
    session.protocol_episode_state = maze::EPISODE_STATE_TERMINAL_REPORTED;
    session.evaluation_state = maze::EVALUATION_STATE_INACTIVE;
    session.episode_state = SessionManager::EpisodeState::Active;
    session.current_episode_id = "training-episode-1";
    session.current_evaluation_id.clear();
    session.current_episode_mode = maze::EPISODE_MODE_TRAINING;
    session.behavior_policy_scope = BehaviorPolicyScope::TrainingFragment;
    session.shortest_action_steps = 2;
    for (int agent_id = 0; agent_id < 2; ++agent_id) {
        SessionManager::AgentRuntime agent;
        agent.done_collected = true;
        agent.reached_goal = true;
        agent.final_termination_reason =
            maze::MAZE_TERMINATION_REASON_GOAL_REACHED;
        agent.episode_return = 10.0;
        agent.episode_transition_count = 2;
        agent.visited.insert(0);
        agent.visited.insert(1);
        agent.reward_component_sums["goal"] = 10.0;
        session.agents.emplace(agent_id, std::move(agent));
    }
}

void TestInitReadinessFailureHasNoCandidateSideEffects() {
    const auto map = MapDescriptor();
    const auto config = Config(map);
    bool fail_receipt = false;
    int receipt_calls = 0;
    std::string published_receipt;
    MazeServiceImpl service(config);
    MazeServiceLifecycleTestAccess::SetModel(service, 0, 0);
    MazeServiceUpdateTestAccess::SetRuntimeReady(service, false);
    MazeServiceLifecycleTestAccess::InstallReceiptWriter(
        service, fail_receipt, receipt_calls, published_receipt);
    auto* session = MazeServiceLifecycleTestAccess::AddSession(service);
    Require(session != nullptr, "readiness fixture session allocated");
    ConfigureInitSession(*session, config);
    const std::string controller_before =
        MazeServiceLifecycleTestAccess::ControllerJson(service);
    maze::InitReq request;
    FillCommand(*session, 1, "init-readiness", request.mutable_command());
    request.mutable_map()->CopyFrom(map);

    MazeServiceLifecycleTestAccess::SetModelAckPending(service, true);
    maze::InitRsp pending;
    service.Init(nullptr, &request, &pending);
    Require(pending.lifecycle().result() == maze::LIFECYCLE_RESULT_REJECTED &&
                receipt_calls == 0 && !session->initialized &&
                session->agents.empty() &&
                session->last_command_sequence == 0 &&
                !session->command_replay.present() &&
                MazeServiceLifecycleTestAccess::ModelAckPending(service) &&
                MazeServiceLifecycleTestAccess::ControllerJson(service) ==
                    controller_before,
            "model ACK pending rejects Init before candidate receipt or commit");

    MazeServiceLifecycleTestAccess::SetModelAckPending(service, false);
    MazeServiceUpdateTestAccess::SetRuntimeReady(service, true);
    maze::InitRsp degraded;
    service.Init(nullptr, &request, &degraded);
    Require(degraded.lifecycle().result() == maze::LIFECYCLE_RESULT_REJECTED &&
                receipt_calls == 0 && !session->initialized &&
                session->map_checksum_sha256.empty() &&
                session->last_command_sequence == 0 &&
                !session->command_replay.present() &&
                MazeServiceUpdateTestAccess::SenderDegraded(service) &&
                MazeServiceLifecycleTestAccess::ControllerJson(service) ==
                    controller_before,
            "degraded sender rejects Init before candidate receipt or commit");

    MazeServiceUpdateTestAccess::SetRuntimeReady(service, false);
    maze::InitRsp applied;
    service.Init(nullptr, &request, &applied);
    Require(applied.lifecycle().result() == maze::LIFECYCLE_RESULT_APPLIED &&
                receipt_calls == 1 && session->initialized &&
                session->last_command_sequence == 1 &&
                session->command_replay.present(),
            "same Init command applies once readiness is restored");
}

void TestInitReceiptFailureIsRetryableAndAtomic() {
    const auto map = MapDescriptor();
    const auto config = Config(map);
    bool fail_receipt = true;
    int receipt_calls = 0;
    std::string published_receipt;
    MazeServiceImpl service(config);
    MazeServiceLifecycleTestAccess::SetModel(service, 0, 0);
    MazeServiceUpdateTestAccess::SetRuntimeReady(service, false);
    MazeServiceLifecycleTestAccess::InstallReceiptWriter(
        service, fail_receipt, receipt_calls, published_receipt);
    auto* session = MazeServiceLifecycleTestAccess::AddSession(service);
    Require(session != nullptr, "Init fixture session allocated");
    ConfigureInitSession(*session, config);

    const std::string controller_before =
        MazeServiceLifecycleTestAccess::ControllerJson(service);
    const auto state_before =
        MazeServiceLifecycleTestAccess::ServiceState(service);
    maze::InitReq request;
    FillCommand(*session, 1, "init-atomic", request.mutable_command());
    request.mutable_map()->CopyFrom(map);

    maze::InitRsp failed;
    service.Init(nullptr, &request, &failed);
    Require(failed.lifecycle().result() == maze::LIFECYCLE_RESULT_REJECTED,
            "receipt failure rejects Init");
    Require(receipt_calls == 1, "failed Init attempted one receipt");
    Require(!session->initialized && session->agents.empty() &&
                session->map_checksum_sha256.empty() &&
                session->blocked.empty() &&
                session->geodesic_distance.empty(),
            "failed Init exposes no candidate map or Agent state");
    Require(session->last_command_sequence == 0 &&
                !session->command_replay.present(),
            "failed Init does not advance command replay");
    Require(session->task_state == maze::TASK_STATE_INITIALIZING &&
                session->session_state == maze::SESSION_STATE_OPENED,
            "failed Init preserves lifecycle state");
    Require(MazeServiceLifecycleTestAccess::ControllerJson(service) ==
                controller_before &&
                !MazeServiceLifecycleTestAccess::ClientInitialized(service),
            "failed Init preserves controller and global init state");
    Require(MazeServiceLifecycleTestAccess::CompletedAgentCount(service) == 0 &&
                MazeServiceLifecycleTestAccess::LastError(service).empty() &&
                MazeServiceLifecycleTestAccess::ServiceState(service) ==
                    state_before,
            "failed Init preserves metrics and service health state");

    fail_receipt = false;
    maze::InitRsp applied;
    service.Init(nullptr, &request, &applied);
    Require(applied.lifecycle().result() == maze::LIFECYCLE_RESULT_APPLIED,
            "same Init command applies after receipt recovery");
    Require(receipt_calls == 2 && session->initialized &&
                session->agents.size() == 2 &&
                session->last_command_sequence == 1 &&
                session->command_replay.present(),
            "successful Init commits map, Agents, and replay once");
    Require(MazeServiceLifecycleTestAccess::ClientInitialized(service) &&
                published_receipt ==
                    MazeServiceLifecycleTestAccess::ControllerJson(service),
            "successful Init publishes and commits the same controller");

    const std::string controller_after =
        MazeServiceLifecycleTestAccess::ControllerJson(service);
    maze::InitRsp replayed;
    service.Init(nullptr, &request, &replayed);
    Require(replayed.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_ALREADY_APPLIED &&
                receipt_calls == 2 &&
                MazeServiceLifecycleTestAccess::ControllerJson(service) ==
                    controller_after &&
                session->last_command_sequence == 1,
            "replayed Init does not repeat receipt or state commit");
}

void TestRejectedInitCanCloseOpenedSession() {
    const auto map = MapDescriptor();
    const auto config = Config(map);
    MazeServiceImpl service(config);
    MazeServiceLifecycleTestAccess::SetModel(service, 0, 0);
    MazeServiceUpdateTestAccess::SetRuntimeReady(service, false);

    auto* session = MazeServiceLifecycleTestAccess::AddSession(service);
    Require(session != nullptr, "Init-close fixture session allocated");
    ConfigureInitSession(*session, config);

    maze::InitReq init_request;
    FillCommand(*session, 1, "init-conclusive-rejection",
                init_request.mutable_command());
    init_request.mutable_map()->CopyFrom(map);
    init_request.mutable_map()->set_map_id("wrong-map-identity");
    maze::InitRsp init_response;
    service.Init(nullptr, &init_request, &init_response);
    Require(init_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_REJECTED &&
                init_response.lifecycle().error_code() ==
                    maze::LIFECYCLE_ERROR_CODE_MAP_INVALID &&
                session->session_state == maze::SESSION_STATE_OPENED &&
                session->episode_state == SessionManager::EpisodeState::None &&
                session->last_command_sequence == 0 &&
                !session->command_replay.present(),
            "conclusive Init rejection preserves an uncommitted OPENED "
            "session");

    maze::CloseSessionReq close_request;
    FillCommand(*session, 1, "close-after-init-rejection",
                close_request.mutable_command());
    maze::CloseSessionRsp close_response;
    service.CloseSession(nullptr, &close_request, &close_response);
    Require(close_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED &&
                session->session_state == maze::SESSION_STATE_CLOSED &&
                session->task_state == maze::TASK_STATE_STOPPED &&
                !session->opened && session->last_command_sequence == 1 &&
                session->command_replay.present(),
            "OPENED session closes with the unconsumed Init sequence");

    maze::CloseSessionRsp replayed;
    service.CloseSession(nullptr, &close_request, &replayed);
    Require(replayed.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_ALREADY_APPLIED &&
                session->last_command_sequence == 1,
            "CloseSession replay remains idempotent after Init rejection");

    auto* active = MazeServiceLifecycleTestAccess::AddSession(service);
    Require(active != nullptr, "active close fixture session allocated");
    ConfigureTrainingEndSession(*active, config);
    maze::CloseSessionReq active_close_request;
    FillCommand(*active, 1, "close-active-episode",
                active_close_request.mutable_command());
    maze::CloseSessionRsp active_close_response;
    service.CloseSession(
        nullptr, &active_close_request, &active_close_response);
    Require(active_close_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_REJECTED &&
                active->session_state ==
                    maze::SESSION_STATE_EPISODE_ACTIVE &&
                active->episode_state ==
                    SessionManager::EpisodeState::Active &&
                active->last_command_sequence == 0 &&
                !active->command_replay.present(),
            "active Episode remains fail-closed for Session cleanup");
}

void TestResumeInitAndBeginUseCheckpointBaseline() {
    const auto map = MapDescriptor();
    auto config = Config(map);
    bool fail_receipt = false;
    int receipt_calls = 0;
    std::string published_receipt;
    MazeServiceImpl service(config);
    MazeServiceLifecycleTestAccess::SetModel(
        service, 201, 200, 10000);
    MazeServiceUpdateTestAccess::SetRuntimeReady(service, false);
    MazeServiceLifecycleTestAccess::InstallReceiptWriter(
        service, fail_receipt, receipt_calls, published_receipt);
    auto* session = MazeServiceLifecycleTestAccess::AddSession(service);
    Require(session != nullptr, "resume fixture session allocated");
    ConfigureInitSession(*session, config);

    maze::InitReq init_request;
    FillCommand(*session, 1, "resume-init", init_request.mutable_command());
    init_request.mutable_map()->CopyFrom(map);
    maze::InitRsp init_response;
    service.Init(nullptr, &init_request, &init_response);
    const auto initialized =
        MazeServiceLifecycleTestAccess::ControllerSnapshot(service);
    Require(init_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED &&
                receipt_calls == 1 && session->initialized &&
                initialized.startup_mode ==
                    SingleMapTaskStartupMode::Resume &&
                initialized.baseline_model_version == 201 &&
                initialized.baseline_train_updates == 200 &&
                initialized.baseline_trained_samples == 10000 &&
                initialized.run_produced_samples == 0,
            "Init captures checkpoint counters without fabricating run samples");
    Require(published_receipt.find("\"startup_mode\":\"resume\"") !=
                std::string::npos &&
                published_receipt.find("\"model_version\":201") !=
                    std::string::npos &&
                published_receipt.find("\"train_updates\":200") !=
                    std::string::npos &&
                published_receipt.find("\"trained_samples\":10000") !=
                    std::string::npos &&
                published_receipt.find("\"run_produced_samples\":0") !=
                    std::string::npos,
            "Init receipt records the resume mode and immutable baseline");

    maze::BeginEpisodeReq begin_request;
    FillCommand(*session, 2, "resume-begin",
                begin_request.mutable_command());
    maze::BeginEpisodeRsp begin_response;
    service.BeginEpisode(nullptr, &begin_request, &begin_response);
    const auto planned =
        MazeServiceLifecycleTestAccess::ControllerSnapshot(service);
    Require(begin_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED &&
                begin_response.assignment().mode() ==
                    maze::EPISODE_MODE_TRAINING &&
                begin_response.assignment().behavior_policy().model_version() ==
                    201 &&
                session->episode_state ==
                    SessionManager::EpisodeState::Active &&
                session->last_command_sequence == 2 &&
                planned.baseline_train_updates == 200 &&
                planned.run_produced_samples == 0,
            "BeginEpisode plans against the full resumed model identity");
}

void TestTrainingEndEpisodeCommitsOnce() {
    const auto map = MapDescriptor();
    const auto config = Config(map);
    bool fail_receipt = false;
    int receipt_calls = 0;
    std::string published_receipt;
    MazeServiceImpl service(config);
    MazeServiceUpdateTestAccess::SetRuntimeReady(service, false);
    MazeServiceLifecycleTestAccess::InstallReceiptWriter(
        service, fail_receipt, receipt_calls, published_receipt);
    auto* session = MazeServiceLifecycleTestAccess::AddSession(service);
    Require(session != nullptr, "EndEpisode fixture session allocated");
    ConfigureTrainingEndSession(*session, config);
    maze::EndEpisodeReq request;
    FillCommand(*session, 1, "end-training", request.mutable_command());
    maze::EndEpisodeRsp applied;
    service.EndEpisode(nullptr, &request, &applied);
    Require(applied.lifecycle().result() == maze::LIFECYCLE_RESULT_APPLIED,
            "training EndEpisode applies");
    Require(receipt_calls == 0 && published_receipt.empty() &&
                MazeServiceLifecycleTestAccess::CompletedAgentCount(service) ==
                    2 &&
                session->last_command_sequence == 1 &&
                session->command_replay.present(),
            "training EndEpisode commits metrics and replay without an "
            "evaluation receipt");
    Require(session->episode_state == SessionManager::EpisodeState::Ended &&
                session->session_state == maze::SESSION_STATE_IDLE &&
                session->protocol_episode_state ==
                    maze::EPISODE_STATE_COMMITTED &&
                session->evaluation_pinned_model_version == -1,
            "training EndEpisode commits terminal lifecycle state");
    maze::EndEpisodeRsp replayed;
    service.EndEpisode(nullptr, &request, &replayed);
    Require(replayed.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_ALREADY_APPLIED &&
                receipt_calls == 0 &&
                MazeServiceLifecycleTestAccess::CompletedAgentCount(service) ==
                    2 &&
                session->last_command_sequence == 1,
            "replayed training EndEpisode does not recount metrics");
}

void TestStatusDoesNotTreatAnOrphanSessionAsAConnectedClient() {
    const auto map = MapDescriptor();
    const auto config = Config(map);
    MazeServiceImpl service(config);
    auto* session = MazeServiceLifecycleTestAccess::AddSession(service);
    Require(session != nullptr, "client activity fixture session allocated");
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

    auto client_recent = [&]() {
        training::AIServerStatusReq request;
        training::AIServerStatusRsp response;
        service.GetAIServerStatus(nullptr, &request, &response);
        for (const auto& value : response.metrics().values()) {
            if (value.field_id() == "server.client.session_recent.v1") {
                return value.value();
            }
        }
        return -1.0;
    };

    MazeServiceLifecycleTestAccess::SetClientActivity(*session, now);
    Require(client_recent() == 1.0,
            "a live lifecycle session reports recent client activity");
    MazeServiceLifecycleTestAccess::SetClientActivity(*session, now - 31000);
    Require(client_recent() == 0.0 &&
                MazeServiceLifecycleTestAccess::ActiveSessionCount(service) ==
                    1,
            "an expired activity lease does not turn an orphan lifecycle "
            "record into a connected Client");

    ConfigureInitSession(*session, config);
    MazeServiceLifecycleTestAccess::SetClientActivity(*session, now - 31000);
    maze::InitReq invalid_request;
    FillCommand(*session, 1, "invalid-client-activity",
                invalid_request.mutable_command());
    invalid_request.mutable_command()->set_lifecycle_epoch(
        session->lifecycle_epoch + 1);
    invalid_request.mutable_map()->CopyFrom(map);
    maze::InitRsp invalid_response;
    service.Init(nullptr, &invalid_request, &invalid_response);
    Require(invalid_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_REJECTED &&
                client_recent() == 0.0,
            "a rejected lifecycle identity cannot renew an orphan Client "
            "activity lease");
}

}  // namespace

int main() {
    TestInitReadinessFailureHasNoCandidateSideEffects();
    TestInitReceiptFailureIsRetryableAndAtomic();
    TestRejectedInitCanCloseOpenedSession();
    TestResumeInitAndBeginUseCheckpointBaseline();
    TestTrainingEndEpisodeCommitsOnce();
    TestStatusDoesNotTreatAnOrphanSessionAsAConnectedClient();
    std::cout << "lifecycle_atomicity_contract: PASS\n";
    return 0;
}
