#include "grpc/maze_service.h"

#include "task/maze_map_contract.h"

#include <cstdlib>
#include <iostream>
#include <string>

struct MazeServiceUpdateTestAccess {
    static void SetRuntimeReady(MazeServiceImpl& service) {
        service.state_.store(training::AISERVER_STATE_READY);
        service.model_state_.store(training::MODEL_STATE_READY);
        std::lock_guard<std::mutex> lock(service.sample_distributor_.mutex_);
        service.sample_distributor_.accepting_ = true;
        service.sample_distributor_.ready_ = true;
        service.sample_distributor_.degraded_ = false;
        service.sample_distributor_.delivery_state_ =
            SampleDistributor::DeliveryState::kHealthy;
        service.sample_distributor_.transient_retry_after_ms_ = 0;
    }
};

struct MazeServiceLifecycleTestAccess {
    static SessionManager::Session* AddSession(MazeServiceImpl& service) {
        const std::string id = service.session_mgr_.CreateSession();
        return service.session_mgr_.GetSession(id);
    }

    static void SetBootstrapModel(MazeServiceImpl& service) {
        service.model_manifest_.model_step = 0;
        service.model_manifest_.sha256 = std::string(64, 'a');
        service.model_manifest_.model_lineage_id = "lifecycle-session";
        service.model_manifest_.manifest_digest = std::string(64, 'f');
        service.model_manifest_.train_updates = 0;
        service.model_manifest_.trained_samples = 0;
        auto* identity = service.model_manifest_.wire.mutable_identity();
        identity->set_model_lineage_id("lifecycle-session");
        identity->set_model_step(0);
        identity->mutable_artifact_digest()->set_algorithm(
            common::DIGEST_ALGORITHM_SHA256);
        identity->mutable_artifact_digest()->set_hex(
            service.model_manifest_.sha256);
        identity->mutable_manifest_digest()->set_algorithm(
            common::DIGEST_ALGORITHM_SHA256);
        identity->mutable_manifest_digest()->set_hex(
            service.model_manifest_.manifest_digest);
        service.model_manifest_.wire.set_train_updates(0);
        service.model_manifest_.wire.set_trained_samples(0);
    }

    static void InstallReceiptWriter(MazeServiceImpl& service, int& calls) {
        service.task_controller_receipt_writer_ =
            [&](const SingleMapTaskController&, std::string&) {
                ++calls;
                return true;
            };
    }

    static SingleMapTaskSnapshot ControllerSnapshot(
        const MazeServiceImpl& service) {
        return service.task_controller_.GetSnapshot();
    }

    static bool LoadActor(MazeServiceImpl& service,
                          const std::string& model_path,
                          std::string& error) {
        return service.onnx_inferencer_.LoadModel(
            model_path, service.config_.model.expected_obs_dim,
            service.config_.model.expected_action_dim, &error);
    }

    static void UseEvaluationUpdateScope(
        MazeServiceImpl& service,
        SessionManager::Session& session) {
        session.current_episode_mode = maze::EPISODE_MODE_EVALUATION;
        session.behavior_policy_scope = BehaviorPolicyScope::EvaluationEpisode;
        session.evaluation_pinned_model_checksum =
            service.model_manifest_.sha256;
    }

    static void SetPendingActionZero(SessionManager::Session& session,
                                     int agent_id,
                                     int gx,
                                     int gy) {
        auto& agent = session.agents.at(agent_id);
        agent.has_pending_action = true;
        agent.pending_action = 0;
        agent.last_action = 0;
        agent.prev_grid_x = gx;
        agent.prev_grid_y = gy;
        agent.observation_grid_x = gx;
        agent.observation_grid_y = gy;
    }

    static void SetGoal(SessionManager::Session& session, int gx, int gy) {
        session.end_gx = gx;
        session.end_gy = gy;
    }

    static int64_t ProducedSamples(const MazeServiceImpl& service) {
        return service.produced_unique_samples_;
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
    Require(error.empty(), "fixture map identity is valid");
    return map;
}

AIServerConfig Config(const maze::MapDescriptor& map) {
    AIServerConfig config;
    config.server.run_mode = aiserver_mode::kTraining;
    config.task.agent_num = 2;
    config.task.fixed_map_id = map.map_id();
    config.task.fixed_map_checksum_sha256 = map.canonical_digest().hex();
    config.task.action_rule_id = map.action_rule_id();
    config.task.shortest_action_steps =
        static_cast<int>(map.shortest_action_steps());
    config.sample_distributor.fragment_samples = 1;
    config.metrics.episode_window = 8;
    config.metrics.event_schema = {
        "maze.metrics.v3", 3,
        {"sha256",
         "34622334da8d4aec593ad231eb0e7cf4465fdee0cbfa13a9ea0e6f864797df73"}};
    return config;
}

void SetTaskIdentity(SessionManager::Session& session,
                     const AIServerConfig& config) {
    session.task.set_task_contract_id(config.task.task_contract_id);
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
    command->set_lifecycle_epoch(session.lifecycle_epoch);
    command->set_command_sequence(sequence);
    command->set_idempotency_key(key);
    command->set_expected_task_state(session.task_state);
    command->set_expected_session_state(session.session_state);
    command->set_expected_episode_state(session.protocol_episode_state);
}

maze::AgentState* AddAgentState(maze::UpdateReq* request,
                                std::uint32_t agent_id,
                                int gx,
                                int gy,
                                bool done,
                                maze::MazeTerminationReason reason,
                                int executed_action = -1) {
    auto* state = request->add_agents();
    state->set_agent_id(agent_id);
    state->mutable_position()->set_x(static_cast<float>(gx));
    state->mutable_position()->set_y(static_cast<float>(gy));
    state->set_is_done(done);
    state->set_termination_reason(reason);
    state->set_last_move_blocked(false);
    if (executed_action >= 0) {
        state->set_executed_action_id(executed_action);
    }
    return state;
}

void RequireRejected(const maze::UpdateRsp& response,
                     const std::string& message) {
    Require(response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_REJECTED &&
                response.lifecycle().ret_code() != 0,
            message);
}

}  // namespace

int main(int argc, char** argv) {
    Require(argc == 2,
            "usage: lifecycle_atomicity_test MODEL");
    const auto map = MapDescriptor();
    const auto config = Config(map);
    MazeServiceImpl service(config);
    MazeServiceLifecycleTestAccess::SetBootstrapModel(service);
    MazeServiceUpdateTestAccess::SetRuntimeReady(service);
    std::string model_error;
    Require(MazeServiceLifecycleTestAccess::LoadActor(
                service, argv[1], model_error),
            "lifecycle Update loads the real ONNX actor: " + model_error);
    int receipt_calls = 0;
    MazeServiceLifecycleTestAccess::InstallReceiptWriter(
        service, receipt_calls);

    auto* session = MazeServiceLifecycleTestAccess::AddSession(service);
    Require(session != nullptr, "session is allocated");
    session->lifecycle_epoch = 7;
    SetTaskIdentity(*session, config);
    session->map_id = config.task.fixed_map_id;
    session->opened = true;
    session->task_state = maze::TASK_STATE_INITIALIZING;
    session->session_state = maze::SESSION_STATE_OPENED;
    session->protocol_episode_state = maze::EPISODE_STATE_UNSPECIFIED;

    maze::InitReq init_request;
    FillCommand(*session, 1, "init", init_request.mutable_command());
    init_request.mutable_map()->CopyFrom(map);
    maze::InitRsp init_response;
    service.Init(nullptr, &init_request, &init_response);
    const auto initialized =
        MazeServiceLifecycleTestAccess::ControllerSnapshot(service);
    Require(init_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED &&
                receipt_calls == 1 && session->initialized &&
                session->agents.size() == 2 &&
                initialized.baseline_model_step == 0 &&
                initialized.baseline_train_updates == 0 &&
                session->last_command_sequence == 1,
            "Init establishes the normal training session");

    maze::BeginEpisodeReq begin_request;
    FillCommand(*session, 2, "begin", begin_request.mutable_command());
    maze::BeginEpisodeRsp begin_response;
    service.BeginEpisode(nullptr, &begin_request, &begin_response);
    Require(begin_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED &&
                begin_response.assignment().mode() ==
                    maze::EPISODE_MODE_TRAINING &&
                begin_response.assignment().behavior_policy().model_step() ==
                    0 &&
                session->episode_state ==
                    SessionManager::EpisodeState::Active &&
                session->last_command_sequence == 2,
            "BeginEpisode binds the bootstrap model to the session");

    MazeServiceLifecycleTestAccess::UseEvaluationUpdateScope(
        service, *session);

    maze::UpdateReq invalid_initial;
    FillCommand(*session, 3, "update-0-invalid",
                invalid_initial.mutable_command());
    invalid_initial.set_frame_id(0);
    AddAgentState(&invalid_initial, 0, 0, 0, false,
                  maze::MAZE_TERMINATION_REASON_ACTIVE, 0);
    AddAgentState(&invalid_initial, 1, 0, 0, false,
                  maze::MAZE_TERMINATION_REASON_ACTIVE);
    maze::UpdateRsp invalid_initial_response;
    service.Update(nullptr, &invalid_initial, &invalid_initial_response);
    RequireRejected(invalid_initial_response,
                    "frame zero rejects an executed action receipt");
    Require(session->last_command_sequence == 2 &&
                session->last_frame_id == -1,
            "rejected initial receipt cannot advance lifecycle state");

    maze::UpdateReq initial_update;
    FillCommand(*session, 3, "update-0", initial_update.mutable_command());
    initial_update.set_frame_id(0);
    AddAgentState(&initial_update, 0, 0, 0, false,
                  maze::MAZE_TERMINATION_REASON_ACTIVE);
    AddAgentState(&initial_update, 1, 0, 0, false,
                  maze::MAZE_TERMINATION_REASON_ACTIVE);
    maze::UpdateRsp initial_update_response;
    service.Update(nullptr, &initial_update, &initial_update_response);
    Require(initial_update_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED &&
                initial_update_response.actions_size() == 2 &&
                session->last_command_sequence == 3 &&
                session->last_frame_id == 0,
            "frame zero accepts every active Agent and returns real actions");

    const int pending_action_0 = session->agents.at(0).pending_action;
    const int pending_action_1 = session->agents.at(1).pending_action;
    const int64_t produced_before_rejections =
        MazeServiceLifecycleTestAccess::ProducedSamples(service);

    maze::UpdateReq missing_receipt;
    FillCommand(*session, 4, "update-1-missing",
                missing_receipt.mutable_command());
    missing_receipt.set_frame_id(1);
    AddAgentState(&missing_receipt, 0, 0, 0, false,
                  maze::MAZE_TERMINATION_REASON_ACTIVE);
    AddAgentState(&missing_receipt, 1, 0, 0, false,
                  maze::MAZE_TERMINATION_REASON_ACTIVE,
                  pending_action_1);
    maze::UpdateRsp missing_receipt_response;
    service.Update(nullptr, &missing_receipt, &missing_receipt_response);
    RequireRejected(missing_receipt_response,
                    "later frame rejects a missing action receipt");

    maze::UpdateReq mismatched_receipt = missing_receipt;
    mismatched_receipt.mutable_command()->set_idempotency_key(
        "update-1-mismatch");
    mismatched_receipt.mutable_agents(0)->set_executed_action_id(
        (pending_action_0 + 1) % 9);
    maze::UpdateRsp mismatched_receipt_response;
    service.Update(nullptr, &mismatched_receipt,
                   &mismatched_receipt_response);
    RequireRejected(mismatched_receipt_response,
                    "later frame rejects a mismatched action receipt");

    maze::UpdateReq out_of_range_receipt = missing_receipt;
    out_of_range_receipt.mutable_command()->set_idempotency_key(
        "update-1-out-of-range");
    out_of_range_receipt.mutable_agents(0)->set_executed_action_id(9);
    maze::UpdateRsp out_of_range_receipt_response;
    service.Update(nullptr, &out_of_range_receipt,
                   &out_of_range_receipt_response);
    RequireRejected(out_of_range_receipt_response,
                    "later frame rejects an out-of-range action receipt");
    Require(session->last_command_sequence == 3 &&
                session->last_frame_id == 0 &&
                MazeServiceLifecycleTestAccess::ProducedSamples(service) ==
                    produced_before_rejections,
            "invalid receipts are rejected before transition/sample commit");

    MazeServiceLifecycleTestAccess::SetPendingActionZero(
        *session, 0, 0, 0);
    MazeServiceLifecycleTestAccess::SetPendingActionZero(
        *session, 1, 1, 0);
    MazeServiceLifecycleTestAccess::SetGoal(*session, 0, 0);
    maze::UpdateReq partial_terminal;
    FillCommand(*session, 4, "update-1-terminal",
                partial_terminal.mutable_command());
    partial_terminal.set_frame_id(1);
    AddAgentState(&partial_terminal, 0, 0, 0, true,
                  maze::MAZE_TERMINATION_REASON_GOAL_REACHED, 0);
    AddAgentState(&partial_terminal, 1, 1, 0, false,
                  maze::MAZE_TERMINATION_REASON_ACTIVE, 0);
    maze::UpdateRsp partial_terminal_response;
    service.Update(nullptr, &partial_terminal,
                   &partial_terminal_response);
    Require(partial_terminal_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED &&
                partial_terminal_response.actions_size() == 1 &&
                partial_terminal_response.actions(0).agent_id() == 1 &&
                session->agents.at(0).done_collected &&
                !session->agents.at(1).done_collected &&
                session->protocol_episode_state ==
                    maze::EPISODE_STATE_RUNNING,
            "first terminal Agent is settled once and receives no next action");

    maze::UpdateReq missing_active;
    FillCommand(*session, 5, "update-2-missing",
                missing_active.mutable_command());
    missing_active.set_frame_id(2);
    maze::UpdateRsp missing_active_response;
    service.Update(nullptr, &missing_active, &missing_active_response);
    RequireRejected(missing_active_response,
                    "Update cannot omit the remaining active Agent");

    maze::UpdateReq terminal_reappears;
    FillCommand(*session, 5, "update-2-reactivate",
                terminal_reappears.mutable_command());
    terminal_reappears.set_frame_id(2);
    AddAgentState(&terminal_reappears, 0, 0, 0, true,
                  maze::MAZE_TERMINATION_REASON_GOAL_REACHED, 0);
    maze::UpdateRsp terminal_reappears_response;
    service.Update(nullptr, &terminal_reappears,
                   &terminal_reappears_response);
    RequireRejected(terminal_reappears_response,
                    "terminal Agent cannot reappear in a later Update");

    maze::UpdateReq unknown_agent;
    FillCommand(*session, 5, "update-2-unknown",
                unknown_agent.mutable_command());
    unknown_agent.set_frame_id(2);
    AddAgentState(&unknown_agent, 9, 1, 0, false,
                  maze::MAZE_TERMINATION_REASON_ACTIVE, 0);
    maze::UpdateRsp unknown_agent_response;
    service.Update(nullptr, &unknown_agent, &unknown_agent_response);
    RequireRejected(unknown_agent_response,
                    "Update rejects an unknown active Agent identity");

    maze::UpdateReq duplicate_agent;
    FillCommand(*session, 5, "update-2-duplicate",
                duplicate_agent.mutable_command());
    duplicate_agent.set_frame_id(2);
    AddAgentState(&duplicate_agent, 1, 1, 0, false,
                  maze::MAZE_TERMINATION_REASON_ACTIVE, 0);
    AddAgentState(&duplicate_agent, 1, 1, 0, false,
                  maze::MAZE_TERMINATION_REASON_ACTIVE, 0);
    maze::UpdateRsp duplicate_agent_response;
    service.Update(nullptr, &duplicate_agent,
                   &duplicate_agent_response);
    RequireRejected(duplicate_agent_response,
                    "Update rejects duplicate active Agent identity");

    MazeServiceLifecycleTestAccess::SetPendingActionZero(
        *session, 1, 1, 0);
    maze::UpdateReq active_only;
    FillCommand(*session, 5, "update-2-active",
                active_only.mutable_command());
    active_only.set_frame_id(2);
    AddAgentState(&active_only, 1, 1, 0, false,
                  maze::MAZE_TERMINATION_REASON_ACTIVE, 0);
    maze::UpdateRsp active_only_response;
    service.Update(nullptr, &active_only, &active_only_response);
    Require(active_only_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED &&
                active_only_response.actions_size() == 1 &&
                active_only_response.actions(0).agent_id() == 1 &&
                session->last_command_sequence == 5 &&
                session->last_frame_id == 2,
            "later Update contains only and exactly every remaining active Agent");
    const int64_t produced_before_replay =
        MazeServiceLifecycleTestAccess::ProducedSamples(service);
    maze::UpdateRsp active_replay;
    service.Update(nullptr, &active_only, &active_replay);
    Require(active_replay.replayed() &&
                active_replay.lifecycle().result() ==
                    maze::LIFECYCLE_RESULT_ALREADY_APPLIED &&
                active_replay.lifecycle().applied_sequence() ==
                    active_only_response.lifecycle().applied_sequence() &&
                active_replay.actions_size() == 1 &&
                active_replay.actions(0).agent_id() == 1 &&
                active_replay.actions(0).SerializeAsString() ==
                    active_only_response.actions(0).SerializeAsString() &&
                MazeServiceLifecycleTestAccess::ProducedSamples(service) ==
                    produced_before_replay,
            "exact Update replay returns the committed result without a duplicate transition");

    MazeServiceLifecycleTestAccess::SetPendingActionZero(
        *session, 1, 1, 0);
    MazeServiceLifecycleTestAccess::SetGoal(*session, 1, 0);
    maze::UpdateReq final_terminal;
    FillCommand(*session, 6, "update-3-terminal",
                final_terminal.mutable_command());
    final_terminal.set_frame_id(3);
    AddAgentState(&final_terminal, 1, 1, 0, true,
                  maze::MAZE_TERMINATION_REASON_GOAL_REACHED, 0);
    maze::UpdateRsp final_terminal_response;
    service.Update(nullptr, &final_terminal,
                   &final_terminal_response);
    Require(final_terminal_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED &&
                final_terminal_response.lifecycle().episode_state() ==
                    maze::EPISODE_STATE_TERMINAL_REPORTED &&
                final_terminal_response.actions_size() == 0 &&
                session->agents.at(0).done_collected &&
                session->agents.at(1).done_collected,
            "final terminal Agent yields an empty action set and terminal lifecycle");

    std::cout << "lifecycle_session_contract: PASS\n";
    return 0;
}
