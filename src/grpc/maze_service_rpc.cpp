#include "grpc/maze_service.h"
#include "rl_sdk/server_command.h"

#include "ai/maze_observation.h"
#include "log/logger.h"
#include "task/maze_map_contract.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_set>
#include "proto/metrics/registry.pb.h"
#include "proto/metrics/transport.pb.h"

namespace {
using rl_sdk::FillReply;
using rl_sdk::CheckCommand;
using rl_sdk::CommitCommand;
using rl_sdk::RejectCommand;
using rl_sdk::CommandCheck;

double ElapsedMs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

void FillServiceIdentity(const std::string& component,
                         const std::string& instance_id,
                         std::uint64_t lifecycle_epoch,
                         common::ServiceInstanceIdentity* target) {
    target->set_component(component);
    target->set_instance_id(instance_id);
    target->set_lifecycle_epoch(lifecycle_epoch);
}

PolicyMode WorkloadModeForRunMode(int run_mode) {
    switch (run_mode) {
        case aiserver_mode::kTraining:
            return PolicyMode::Training;
        case aiserver_mode::kEvaluation:
            return PolicyMode::Evaluation;
        default:
            return PolicyMode::Unspecified;
    }
}

bool IsEnvironmentTerminal(maze::MazeTerminationReason reason) {
    return reason == maze::MAZE_TERMINATION_REASON_GOAL_REACHED ||
           reason == maze::MAZE_TERMINATION_REASON_TIME_LIMIT;
}

void FillOpenRejected(rl::session::v1::CommandErrorCode error_code,
                      const std::string& message,
                      rl::session::v1::CommandReply* reply) {
    reply->set_result(rl::session::v1::COMMAND_RESULT_REJECTED);
    reply->set_error_code(error_code);
    reply->set_message(message);
    reply->set_applied_sequence(0);
    reply->set_phase(rl::session::v1::SESSION_PHASE_UNSPECIFIED);
}


}  // namespace

grpc::Status MazeServiceImpl::OpenSession(
    grpc::ServerContext*,
    const maze::OpenSessionReq* req,
    maze::OpenSessionRsp* rsp) {
    std::lock_guard<std::mutex> lock(runtime_.mutex_);
    const std::string payload = req->SerializeAsString();
    const auto prior = runtime_.open_payloads_.find(req->request_id());
    if (prior != runtime_.open_payloads_.end()) {
        if (prior->second != payload) {
            FillOpenRejected(
                rl::session::v1::COMMAND_ERROR_CODE_PAYLOAD_CONFLICT,
                "OpenSession request_id conflicts", rsp->mutable_reply());
            return grpc::Status::OK;
        }
        const auto cached = runtime_.open_responses_.find(req->request_id());
        if (cached != runtime_.open_responses_.end() &&
            rsp->ParseFromString(cached->second)) {
            auto* session = runtime_.session_mgr_.GetSession(rsp->session_id());
            if (session != nullptr) {
                session->last_valid_client_activity_unix_ms = runtime_.NowMs();
            }
            rsp->mutable_reply()->set_result(
                rl::session::v1::COMMAND_RESULT_ALREADY_APPLIED);
            rsp->mutable_reply()->set_message(
                "session was already opened");
            return grpc::Status::OK;
        }
        FillOpenRejected(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                         "OpenSession response is unavailable",
                         rsp->mutable_reply());
        return grpc::Status::OK;
    }

    if (!runtime_.IsReady()) {
        FillOpenRejected(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                         "AIServer is not ready", rsp->mutable_reply());
        return grpc::Status::OK;
    }
    if (req->request_id().empty() ||
        req->client().component().empty() ||
        req->client().instance_id().empty() ||
        req->client().lifecycle_epoch() == 0 ||
        req->environment_instance_id().empty()) {
        FillOpenRejected(rl::session::v1::COMMAND_ERROR_CODE_INVALID_IDENTITY,
                         "Client or environment identity is invalid",
                         rsp->mutable_reply());
        return grpc::Status::OK;
    }

    const std::string session_id = runtime_.session_mgr_.CreateSession();
    auto* session = runtime_.session_mgr_.GetSession(session_id);
    if (!session) {
        FillOpenRejected(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                         "AIServer could not allocate a session",
                         rsp->mutable_reply());
        return grpc::Status::OK;
    }
    session->client.CopyFrom(req->client());
    session->environment_instance_id = req->environment_instance_id();
    session->last_valid_client_activity_unix_ms = runtime_.NowMs();
    session->session_epoch = runtime_.next_lifecycle_epoch_.fetch_add(1);
    session->task.map_id = task_adapter_.Config().task.fixed_map_id;
    session->workload_mode = WorkloadModeForRunMode(task_adapter_.Config().server.run_mode);
    session->phase = rl::session::v1::SESSION_PHASE_OPEN;

    rsp->set_session_id(session_id);
    rsp->set_session_epoch(session->session_epoch);
    FillServiceIdentity("rl-aiserver", runtime_.producer_instance_id_,
                        runtime_.producer_lifecycle_epoch_,
                        rsp->mutable_aiserver());
    auto* environment = rsp->mutable_environment();
    environment->set_agent_count(
        static_cast<std::uint32_t>(task_adapter_.Config().environment.agent_count));
    environment->set_map_id(task_adapter_.Config().task.fixed_map_id);
    environment->set_episode_max_steps(
        static_cast<std::uint32_t>(task_adapter_.Config().task.episode_max_steps));
    environment->set_action_mask_mode(
        task_adapter_.Config().policy.action_mask_mode == "required"
            ? maze::ACTION_MASK_MODE_REQUIRED
            : maze::ACTION_MASK_MODE_DISABLED);
    rsp->set_workload_mode(session->workload_mode == PolicyMode::Training
        ? maze::WORKLOAD_MODE_TRAINING : maze::WORKLOAD_MODE_EVALUATION);
    FillReply(*session, 0, rl::session::v1::COMMAND_RESULT_APPLIED,
              rl::session::v1::COMMAND_ERROR_CODE_UNSPECIFIED,
              "session opened", rsp->mutable_reply());
    runtime_.open_payloads_[req->request_id()] = payload;
    runtime_.open_responses_[req->request_id()] = rsp->SerializeAsString();
    LOG_INFO("MazeService", "Session 已分配: %s", session_id.c_str());
    return grpc::Status::OK;
}

grpc::Status MazeServiceImpl::Init(
    grpc::ServerContext*,
    const maze::InitReq* req,
    maze::InitRsp* rsp) {
    std::lock_guard<std::mutex> lock(runtime_.mutex_);
    auto* session = runtime_.session_mgr_.GetSession(req->command().session_id());
    if (!session) {
        FillOpenRejected(rl::session::v1::COMMAND_ERROR_CODE_INVALID_IDENTITY,
                         "session does not exist", rsp->mutable_reply());
        return grpc::Status::OK;
    }
    const auto check = CheckCommand(*session, req->command(), *req, rsp);
    if (check == CommandCheck::Proceed || check == CommandCheck::Replayed) {
        session->last_valid_client_activity_unix_ms = runtime_.NowMs();
    }
    if (check != CommandCheck::Proceed) return grpc::Status::OK;
    if (session->phase != rl::session::v1::SESSION_PHASE_OPEN ||
        !req->command().episode_id().empty()) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      "Init is not valid in the current session state",
                      rsp->mutable_reply());
        return grpc::Status::OK;
    }
    if (!runtime_.IsReady() || runtime_.model_ack_pending_) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      "AIServer is not ready to initialize the assigned task",
                      rsp->mutable_reply());
        return grpc::Status::OK;
    }

    ValidatedMazeMap validated;
    std::string error;
    if (!ValidateMazeMapDescriptor(req->map(), task_adapter_.Config().task.fixed_map_id,
                                   validated, error)) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_TASK_INPUT_INVALID,
                      error.empty() ? "map descriptor does not match assignment"
                                    : error,
                      rsp->mutable_reply());
        return grpc::Status::OK;
    }

    SessionManager::Session candidate = *session;
    SingleMapTaskController candidate_task_controller = task_controller_;
    candidate.task.map_id = req->map().map_id();
    candidate.task.shortest_action_steps = validated.shortest_action_steps;
    candidate.task.grid_cols = static_cast<int>(req->map().grid_columns());
    candidate.task.grid_rows = static_cast<int>(req->map().grid_rows());
    candidate.task.grid_size_microunits = req->map().grid_size_microunits();
    candidate.task.start_gx = req->map().start_grid_x();
    candidate.task.start_gy = req->map().start_grid_y();
    candidate.task.end_gx = req->map().goal_grid_x();
    candidate.task.end_gy = req->map().goal_grid_y();
    candidate.task.blocked = std::move(validated.blocked);
    candidate.task.geodesic_distance = std::move(validated.geodesic_distance);
    candidate.task.max_finite_geodesic_distance = validated.max_finite_distance;
    candidate.agents.clear();
    for (int agent_id = 0; agent_id < task_adapter_.Config().environment.agent_count;
         ++agent_id) {
        candidate.agents[agent_id];
    }
    if (task_adapter_.Config().server.run_mode == aiserver_mode::kTraining) {
        if (!candidate_task_controller.Initialize(
                task_adapter_.Config().task.episode_max_steps, ActiveModelIdentity(),
                runtime_.produced_unique_transitions_, error) ||
            !WriteTaskControllerReceipt(candidate_task_controller, error)) {
            RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "TaskController initialization failed: " + error,
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }
    }
    candidate.phase = rl::session::v1::SESSION_PHASE_READY;
    CommitCommand(candidate, req->command(), *req, rsp,
                  "map accepted and session initialized");
    if (task_adapter_.Config().server.run_mode == aiserver_mode::kTraining) {
        task_controller_ = std::move(candidate_task_controller);
    }
    *session = std::move(candidate);
    runtime_.client_initialized_ = true;
    return grpc::Status::OK;
}

grpc::Status MazeServiceImpl::BeginEpisode(
    grpc::ServerContext*,
    const maze::BeginEpisodeReq* req,
    maze::BeginEpisodeRsp* rsp) {
    std::lock_guard<std::mutex> lock(runtime_.mutex_);
    auto* session = runtime_.session_mgr_.GetSession(req->command().session_id());
    if (!session) {
        FillOpenRejected(rl::session::v1::COMMAND_ERROR_CODE_INVALID_IDENTITY,
                         "session does not exist", rsp->mutable_reply());
        return grpc::Status::OK;
    }
    const auto check = CheckCommand(*session, req->command(), *req, rsp);
    if (check == CommandCheck::Proceed || check == CommandCheck::Replayed) {
        session->last_valid_client_activity_unix_ms = runtime_.NowMs();
    }
    if (check != CommandCheck::Proceed) return grpc::Status::OK;
    if (!runtime_.IsCoreInferenceReady()) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      "AIServer core inference is not ready to begin an episode",
                      rsp->mutable_reply());
        return grpc::Status::OK;
    }
    if (session->phase != rl::session::v1::SESSION_PHASE_READY ||
        !req->command().episode_id().empty()) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      "BeginEpisode is not valid in the current state",
                      rsp->mutable_reply());
        return grpc::Status::OK;
    }

    if (task_adapter_.Config().server.run_mode == aiserver_mode::kTraining) {
        const auto sender = runtime_.sample_distributor_.GetSnapshot();
        const bool recoverable_delivery_state =
            sender.delivery_state ==
                SampleDistributor::DeliveryState::kHealthy ||
            sender.delivery_state ==
                SampleDistributor::DeliveryState::kLocalBackpressure ||
            sender.delivery_state ==
                SampleDistributor::DeliveryState::kTransientRetry;
        if (sender.terminal_fault || sender.degraded || !sender.ready ||
            !recoverable_delivery_state) {
            RejectCommand(
                *session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                sender.last_error.empty()
                    ? "sample delivery cannot begin an episode"
                    : "sample delivery cannot begin an episode: " +
                          sender.last_error,
                rsp->mutable_reply());
            return grpc::Status::OK;
        }
    }

    SingleMapEpisodePlan plan;
    SingleMapTaskController candidate_task_controller = task_controller_;
    ModelManifest planned_model = runtime_.model_manifest_;
    SingleMapModelIdentity planned_model_identity = ActiveModelIdentity();
    std::string error;
    if (task_adapter_.Config().server.run_mode == aiserver_mode::kTraining) {
        if (!candidate_task_controller.PlanNextEpisode(
                planned_model_identity, runtime_.produced_unique_transitions_,
                plan, error)) {
            runtime_.MarkDegraded("TaskController planning failed: " + error);
            RejectCommand(*session,
                          rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          runtime_.last_error_, rsp->mutable_reply());
            return grpc::Status::OK;
        }
        if (plan.episode_mode != maze::EPISODE_MODE_TRAINING) {
            runtime_.MarkDegraded(
                "training TaskController attempted to schedule evaluation");
            RejectCommand(*session,
                          rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          runtime_.last_error_, rsp->mutable_reply());
            return grpc::Status::OK;
        }
    } else {
        plan.episode_mode = maze::EPISODE_MODE_EVALUATION;
        plan.max_steps = task_adapter_.Config().task.episode_max_steps;
        plan.model = ActiveModelIdentity();
    }

    SessionManager::Session candidate = *session;
    maze::EpisodeAssignment assignment;
    const bool training_workload =
        task_adapter_.Config().server.run_mode == aiserver_mode::kTraining;
    const bool training_identity_matches =
        !training_workload ||
        (plan.model.model_step == planned_model.model_step() &&
         plan.model.trained_samples == planned_model.trained_samples() &&
         plan.model.model_lineage_id == planned_model.model_lineage_id() &&
         !planned_model.model_lineage_id().empty());
    if (plan.max_steps <= 0 ||
        !training_identity_matches) {
        RejectCommand(*session,
                      rl::session::v1::COMMAND_ERROR_CODE_MODEL_IDENTITY_MISMATCH,
                      "Episode plan does not match the loaded model identity",
                      rsp->mutable_reply());
        return grpc::Status::OK;
    }

    const uint64_t episode_sequence = runtime_.next_episode_id_.load();
    const std::string episode_id =
        "maze-episode-" + std::to_string(episode_sequence);
    candidate.current_episode_mode = plan.episode_mode == maze::EPISODE_MODE_TRAINING
        ? PolicyMode::Training : PolicyMode::Evaluation;
    candidate.task.current_max_steps = plan.max_steps;
    const bool training_episode =
        plan.episode_mode == maze::EPISODE_MODE_TRAINING;
    candidate.behavior_policy_scope =
        training_episode
            ? BehaviorPolicyScope::TrainingAgentSegment
            : BehaviorPolicyScope::EvaluationEpisode;
    candidate.evaluation_pinned_model_lineage_id.clear();
    candidate.evaluation_pinned_model_step = 0;
    if (!training_episode) {
        candidate.evaluation_pinned_model_lineage_id =
            plan.model.model_lineage_id;
        candidate.evaluation_pinned_model_step = plan.model.model_step;
    }
    task_adapter_.ResetEpisodeState(candidate, episode_id);
    candidate.phase = rl::session::v1::SESSION_PHASE_EPISODE_RUNNING;
    assignment.set_episode_id(episode_id);
    assignment.set_mode(plan.episode_mode);
    assignment.set_max_steps(static_cast<std::uint32_t>(plan.max_steps));
    rsp->mutable_assignment()->CopyFrom(assignment);
    CommitCommand(
        candidate, req->command(), *req, rsp,
        training_episode
            ? "training episode assigned with per-Agent segment model pinning"
            : "evaluation episode assigned with episode-pinned behavior policy");
    if (task_adapter_.Config().server.run_mode == aiserver_mode::kTraining) {
        task_controller_ = std::move(candidate_task_controller);
    }
    *session = std::move(candidate);
    runtime_.next_episode_id_.store(episode_sequence + 1);
    return grpc::Status::OK;
}

grpc::Status MazeServiceImpl::Update(
    grpc::ServerContext*,
    const maze::UpdateReq* req,
    maze::UpdateRsp* rsp) {
    const auto rpc_start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(runtime_.mutex_);
    auto finish = [&]() { runtime_.RecordUpdateLatency(rpc_start); };
    auto* session = runtime_.session_mgr_.GetSession(req->command().session_id());
    if (!session) {
        FillOpenRejected(rl::session::v1::COMMAND_ERROR_CODE_INVALID_IDENTITY,
                         "session does not exist", rsp->mutable_reply());
        finish();
        return grpc::Status::OK;
    }
    const auto check = CheckCommand(*session, req->command(), *req, rsp);
    if (check == CommandCheck::Proceed || check == CommandCheck::Replayed) {
        session->last_valid_client_activity_unix_ms = runtime_.NowMs();
    }
    if (check != CommandCheck::Proceed) {
        finish();
        return grpc::Status::OK;
    }
    if (session->phase != rl::session::v1::SESSION_PHASE_EPISODE_RUNNING) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      "Update requires a running active Episode",
                      rsp->mutable_reply());
        finish();
        return grpc::Status::OK;
    }
    if (req->command().episode_id() != session->current_episode_id) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_INVALID_IDENTITY,
                      "Update Episode identity does not match",
                      rsp->mutable_reply());
        finish();
        return grpc::Status::OK;
    }
    const auto active_agent_count = static_cast<int>(std::count_if(
        session->agents.begin(), session->agents.end(),
        [](const auto& item) { return !item.second.done_collected; }));
    if (req->frame_id() !=
            static_cast<std::uint64_t>(session->last_frame_id + 1) ||
        req->agents_size() != active_agent_count) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_OUT_OF_ORDER,
                      "frame is not contiguous or does not contain every "
                      "active Agent exactly once",
                      rsp->mutable_reply());
        finish();
        return grpc::Status::OK;
    }
    if (session->task.current_max_steps <= 0 ||
        req->frame_id() >
            static_cast<std::uint64_t>(session->task.current_max_steps)) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_OUT_OF_ORDER,
                      "frame exceeds the assigned Episode horizon",
                      rsp->mutable_reply());
        finish();
        return grpc::Status::OK;
    }
    if (!runtime_.IsCoreInferenceReady()) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      "AIServer core inference is not ready",
                      rsp->mutable_reply());
        finish();
        return grpc::Status::OK;
    }

    const bool training_episode =
        session->current_episode_mode == PolicyMode::Training;
    const BehaviorPolicyScope expected_policy_scope =
        training_episode
            ? BehaviorPolicyScope::TrainingAgentSegment
            : BehaviorPolicyScope::EvaluationEpisode;
    if (session->behavior_policy_scope != expected_policy_scope) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      "Episode mode and behavior policy scope disagree",
                      rsp->mutable_reply());
        finish();
        return grpc::Status::OK;
    }
    if (!training_episode &&
        (session->evaluation_pinned_model_lineage_id !=
             runtime_.model_manifest_.model_lineage_id() ||
         session->evaluation_pinned_model_step !=
             runtime_.model_manifest_.model_step())) {
        RejectCommand(*session,
                      rl::session::v1::COMMAND_ERROR_CODE_MODEL_IDENTITY_MISMATCH,
                      "evaluation behavior policy identity changed",
                      rsp->mutable_reply());
        finish();
        return grpc::Status::OK;
    }

    const bool collect =
        task_adapter_.Config().server.run_mode == aiserver_mode::kTraining &&
        training_episode;
    if (collect) {
        const auto sender = runtime_.sample_distributor_.GetSnapshot();
        if (sender.terminal_fault || sender.degraded) {
            RejectCommand(
                *session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                sender.last_error.empty()
                    ? "SampleDistributor has a terminal fault"
                    : "SampleDistributor has a terminal fault: " +
                          sender.last_error,
                rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }
        if (sender.sample_delivery_paused ||
            sender.delivery_state !=
                SampleDistributor::DeliveryState::kHealthy) {
            rsp->mutable_wait()->set_retry_after_ms(
                runtime_.sample_distributor_.PauseRetryAfterMs());
            FillReply(
                *session, session->last_command_sequence,
                rl::session::v1::COMMAND_RESULT_WAIT,
                rl::session::v1::COMMAND_ERROR_CODE_UNSPECIFIED,
                sender.transient_retry
                    ? "SampleDistributor is recovering its ingress transport"
                    : "SampleDistributor local outbound queue is full",
                rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }
        if (!sender.ready) {
            RejectCommand(*session,
                          rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "SampleDistributor is not ready",
                          rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }
    }

    SessionManager::Session candidate = *session;
    int64_t candidate_produced_transitions =
        runtime_.produced_unique_transitions_;
    int64_t candidate_produced_envelopes =
        runtime_.produced_unique_envelopes_;
    auto candidate_produced_by_model = runtime_.produced_transitions_by_model_;
    uint64_t candidate_segment_sequence = runtime_.next_segment_seq_.load();
    int64_t candidate_model_activation_count =
        runtime_.per_agent_model_activation_count_;
    bool candidate_latest_model_used =
        runtime_.latest_prepared_used_by_agent_;
    int64_t candidate_closed_segment_count =
        runtime_.closed_segment_count_;
    auto candidate_close_counts = runtime_.segment_close_counts_;
    SingleMapTaskController candidate_task_controller = task_controller_;
    std::mt19937 candidate_action_rng = runtime_.action_rng_;
    std::vector<training::ProcessedTransitionEnvelope> prepared_envelopes;
    std::vector<maze::AgentAction> prepared_actions;

    std::unordered_set<int> seen;
    for (const auto& state : req->agents()) {
        const int agent_id = static_cast<int>(state.agent_id());
        auto agent_it = candidate.agents.find(agent_id);
        if (agent_it == candidate.agents.end() ||
            !seen.insert(agent_id).second ||
            !std::isfinite(state.position().x()) ||
            !std::isfinite(state.position().y()) ||
            std::trunc(state.position().x()) != state.position().x() ||
            std::trunc(state.position().y()) != state.position().y()) {
            RejectCommand(*session,
                          rl::session::v1::COMMAND_ERROR_CODE_INVALID_IDENTITY,
                          "Agent identity or position is invalid",
                          rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }
        auto& agent = agent_it->second;
        if (agent.done_collected) {
            RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "terminal Agent cannot be reported again",
                          rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }

        const bool mask_required =
            task_adapter_.Config().policy.action_mask_mode == "required";
        const bool mask_has_available =
            std::any_of(state.action_mask().begin(),
                        state.action_mask().end(),
                        [](bool available) { return available; });
        if ((state.is_done() && state.action_mask_size() != 0) ||
            (!state.is_done() && mask_required &&
             (state.action_mask_size() !=
                  task_adapter_.Config().model.expected_action_dim ||
              !mask_has_available)) ||
            (!state.is_done() && !mask_required &&
             state.action_mask_size() != 0)) {
            RejectCommand(*session,
                          rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "Agent action mask contradicts the session mode",
                          rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }

        const bool initial_observation = req->frame_id() == 0;
        if ((initial_observation && state.has_executed_action_id()) ||
            (!initial_observation &&
             (!state.has_executed_action_id() ||
              state.executed_action_id() < 0 ||
              state.executed_action_id() > 8 ||
              !agent.has_pending_action ||
              state.executed_action_id() != agent.pending_action))) {
            RejectCommand(
                *session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                initial_observation
                    ? "initial Agent state cannot report an executed action"
                    : "executed action receipt does not match the pending "
                      "AIServer action",
                rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }

        const int gx = static_cast<int>(state.position().x());
        const int gy = static_cast<int>(state.position().y());
        const bool goal =
            state.termination_reason() ==
            maze::MAZE_TERMINATION_REASON_GOAL_REACHED;
        const bool timeout =
            state.termination_reason() ==
            maze::MAZE_TERMINATION_REASON_TIME_LIMIT;
        if (!candidate.task.IsWalkable(gx, gy) ||
            state.is_done() !=
                IsEnvironmentTerminal(state.termination_reason()) ||
            (!state.is_done() &&
             state.termination_reason() !=
                 maze::MAZE_TERMINATION_REASON_ACTIVE) ||
            (!state.is_done() &&
             req->frame_id() >= static_cast<std::uint64_t>(
                                        candidate.task.current_max_steps)) ||
            (timeout &&
             req->frame_id() != static_cast<std::uint64_t>(
                                        candidate.task.current_max_steps)) ||
            (!agent.has_pending_action &&
             agent.last_observation_frame_id < 0 && state.is_done())) {
            RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "Agent state or termination reason is invalid",
                          rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }
        std::string observation_error;
        if (!MazeObservation::ApplyState(
                candidate, agent, gx, gy,
                static_cast<int64_t>(req->frame_id()), state.is_done(),
                state.last_move_blocked(), observation_error)) {
            RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          observation_error, rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }

        std::string transition_error;
        if (!FinalizePendingTransition(
                candidate, agent_id, gx, gy, state.is_done(),
                state.termination_reason(), collect, transition_error)) {
            ++runtime_.rollout_estimator_failure_count_;
            runtime_.MarkDegraded("Episode transition preparation failed: " +
                         transition_error);
            RejectCommand(*session,
                          rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          runtime_.last_error_, rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }
        if (state.is_done()) {
            agent.task.reached_goal = goal;
            agent.done_collected = true;
            agent.task.final_termination_reason = state.termination_reason();
            agent.terminal_frame_id =
                static_cast<int64_t>(req->frame_id());
        }
    }

    std::string prepare_error;
    if (collect) {
        for (const auto& state : req->agents()) {
            const int agent_id = static_cast<int>(state.agent_id());
            auto& agent = candidate.agents.at(agent_id);
            if (!agent.segment_open || agent.segment_transitions.empty()) {
                continue;
            }

            training::SegmentCloseReason close_reason =
                training::SEGMENT_CLOSE_REASON_UNSPECIFIED;
            float bootstrap_value = 0.0f;
            bool bootstrap_applied = false;
            if (state.is_done()) {
                close_reason =
                    training::SEGMENT_CLOSE_REASON_ENVIRONMENT_TERMINATED;
            } else {
                const auto tmax = task_adapter_.Config().rollout.tmax;
                if (agent.segment_transitions.size() >
                    static_cast<std::size_t>(tmax)) {
                    ++runtime_.rollout_estimator_failure_count_;
                    runtime_.MarkDegraded(
                        "Agent segment exceeded its configured TMax");
                    RejectCommand(*session,
                                  rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                                  runtime_.last_error_, rsp->mutable_reply());
                    finish();
                    return grpc::Status::OK;
                }
                if (agent.segment_transitions.size() !=
                    static_cast<std::size_t>(tmax)) {
                    continue;
                }
                close_reason = training::SEGMENT_CLOSE_REASON_TMAX;
                bootstrap_applied = true;
                if (!runtime_.InferPinnedValue(
                        agent,
                        agent.segment_transitions.back().next_observation,
                        bootstrap_value)) {
                    ++runtime_.rollout_estimator_failure_count_;
                    RejectCommand(*session,
                                  rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                                  runtime_.last_error_, rsp->mutable_reply());
                    finish();
                    return grpc::Status::OK;
                }
            }

            if (!runtime_.PrepareAgentSegmentClose(
                    candidate, agent_id, close_reason,
                    bootstrap_value, bootstrap_applied,
                    candidate_produced_transitions,
                    candidate_produced_envelopes,
                    candidate_produced_by_model, prepared_envelopes,
                    candidate_close_counts, prepare_error)) {
                ++runtime_.rollout_estimator_failure_count_;
                runtime_.MarkDegraded("Agent segment close failed: " + prepare_error);
                RejectCommand(*session,
                              rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                              runtime_.last_error_, rsp->mutable_reply());
                finish();
                return grpc::Status::OK;
            }
            ++candidate_closed_segment_count;
        }
    }

    const bool all_done = std::all_of(
        candidate.agents.begin(), candidate.agents.end(),
        [](const auto& item) { return item.second.done_collected; });
    candidate.last_frame_id = static_cast<int64_t>(req->frame_id());
    std::string commit_message;
    if (all_done) {
        candidate.phase = rl::session::v1::SESSION_PHASE_EPISODE_TERMINAL;
        rsp->mutable_action_batch();
        commit_message = "terminal Agent states accepted";
    } else {
        for (const auto& state : req->agents()) {
            const int agent_id = static_cast<int>(state.agent_id());
            auto& agent = candidate.agents.at(agent_id);
            if (state.is_done()) continue;
            int action = 0;
            float log_probability = 0.0f;
            float value = 0.0f;
            const std::vector<bool> action_mask(
                state.action_mask().begin(), state.action_mask().end());
            if (!PrepareModelAction(
                    candidate, agent, agent_id,
                    static_cast<int>(state.position().x()),
                    static_cast<int>(state.position().y()),
                    static_cast<int64_t>(req->frame_id()),
                    action_mask,
                    candidate_segment_sequence,
                    candidate_model_activation_count,
                    candidate_latest_model_used,
                    candidate_action_rng, action,
                    log_probability, value)) {
                RejectCommand(*session,
                              rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                              runtime_.last_error_, rsp->mutable_reply());
                finish();
                return grpc::Status::OK;
            }
            maze::AgentAction response_action;
            response_action.set_agent_id(
                static_cast<std::uint32_t>(agent_id));
            response_action.set_action_id(
                static_cast<maze::MazeAction>(action));
            prepared_actions.push_back(std::move(response_action));
        }
        for (const auto& response_action : prepared_actions) {
            *rsp->mutable_action_batch()->add_actions() = response_action;
        }
        commit_message = "Agent states accepted and actions assigned";
    }

    if (training_episode) {
        SingleMapModelIdentity latest_identity = ActiveModelIdentity();
        std::string controller_error;
        if (!candidate_task_controller.ObserveTrainingProgress(
                latest_identity, candidate_produced_transitions,
                controller_error)) {
            runtime_.MarkDegraded("TaskController collection check failed: " +
                         controller_error);
            RejectCommand(*session,
                          rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          runtime_.last_error_, rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }
    }

    CommitCommand(candidate, req->command(), *req, rsp, commit_message);

    const auto wait_prepared = [&](const std::string& message) {
        rsp->Clear();
        rsp->mutable_wait()->set_retry_after_ms(
            runtime_.sample_distributor_.PauseRetryAfterMs());
        FillReply(*session, session->last_command_sequence,
                      rl::session::v1::COMMAND_RESULT_WAIT,
                      rl::session::v1::COMMAND_ERROR_CODE_UNSPECIFIED,
                      message, rsp->mutable_reply());
    };
    const auto reject_prepared = [&](const std::string& message) {
        rsp->Clear();
        RejectCommand(*session,
                      rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      message, rsp->mutable_reply());
    };

    uint64_t enqueue_reservation = 0;
    const auto enqueue_start = std::chrono::steady_clock::now();
    const auto reservation_result =
        runtime_.sample_distributor_.ReserveEnqueueEnvelopeSet(
            prepared_envelopes, enqueue_reservation, prepare_error);
    if (reservation_result ==
        SampleDistributor::ReservationResult::kRetryableUnavailable) {
        wait_prepared(
            prepare_error.empty()
                ? "SampleDistributor capacity changed before frame commit"
                : prepare_error);
        finish();
        return grpc::Status::OK;
    }
    if (reservation_result ==
        SampleDistributor::ReservationResult::kTerminalFault) {
        runtime_.MarkDegraded("failed to reserve processed envelope set: " +
                     prepare_error);
        reject_prepared(runtime_.last_error_);
        finish();
        return grpc::Status::OK;
    }

    const auto seal_result =
        runtime_.sample_distributor_.SealEnqueueEnvelopeSet(
            enqueue_reservation, prepare_error);
    if (seal_result ==
        SampleDistributor::SealResult::kRetryableUnavailable) {
        wait_prepared(
            prepare_error.empty()
                ? "SampleDistributor state changed before frame seal"
                : prepare_error);
        finish();
        return grpc::Status::OK;
    }
    if (seal_result == SampleDistributor::SealResult::kTerminalFault) {
        runtime_.MarkDegraded("failed to seal processed envelope set: " +
                     prepare_error);
        reject_prepared(runtime_.last_error_);
        finish();
        return grpc::Status::OK;
    }

    const auto enqueue_commit =
        runtime_.sample_distributor_.CommitEnqueueEnvelopeSet(
            enqueue_reservation, prepare_error);
    if (enqueue_commit != SampleDistributor::CommitResult::kCommitted) {
        runtime_.MarkDegraded(
            "sealed processed envelope-set commit invariant failed: " +
            prepare_error);
        reject_prepared(runtime_.last_error_);
        finish();
        return grpc::Status::OK;
    }

    if (!prepared_envelopes.empty()) {
        const double latency_ms = ElapsedMs(enqueue_start);
        runtime_.enqueue_count_ +=
            static_cast<int64_t>(prepared_envelopes.size());
        runtime_.enqueue_latency_sum_ms_ += latency_ms;
        runtime_.enqueue_latency_max_ms_ =
            std::max(runtime_.enqueue_latency_max_ms_, latency_ms);
    }
    *session = std::move(candidate);
    runtime_.produced_unique_transitions_ = candidate_produced_transitions;
    runtime_.produced_unique_envelopes_ = candidate_produced_envelopes;
    runtime_.produced_transitions_by_model_ =
        std::move(candidate_produced_by_model);
    runtime_.next_segment_seq_.store(candidate_segment_sequence);
    runtime_.per_agent_model_activation_count_ =
        candidate_model_activation_count;
    runtime_.latest_prepared_used_by_agent_ =
        candidate_latest_model_used;
    runtime_.closed_segment_count_ = candidate_closed_segment_count;
    runtime_.segment_close_counts_ = std::move(candidate_close_counts);
    task_controller_ = std::move(candidate_task_controller);
    runtime_.action_rng_ = std::move(candidate_action_rng);
    finish();
    return grpc::Status::OK;
}

grpc::Status MazeServiceImpl::EndEpisode(
    grpc::ServerContext*,
    const maze::EndEpisodeReq* req,
    maze::EndEpisodeRsp* rsp) {
    std::lock_guard<std::mutex> lock(runtime_.mutex_);
    auto* session = runtime_.session_mgr_.GetSession(req->command().session_id());
    if (!session) {
        FillOpenRejected(rl::session::v1::COMMAND_ERROR_CODE_INVALID_IDENTITY,
                         "session does not exist", rsp->mutable_reply());
        return grpc::Status::OK;
    }
    const auto check = CheckCommand(*session, req->command(), *req, rsp);
    if (check == CommandCheck::Proceed || check == CommandCheck::Replayed) {
        session->last_valid_client_activity_unix_ms = runtime_.NowMs();
    }
    if (check != CommandCheck::Proceed) return grpc::Status::OK;
    if (runtime_.shutdown_started_) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      "AIServer is draining and no longer accepts EndEpisode",
                      rsp->mutable_reply());
        return grpc::Status::OK;
    }
    if (session->phase != rl::session::v1::SESSION_PHASE_EPISODE_TERMINAL ||
        req->command().episode_id() != session->current_episode_id) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      "EndEpisode requires a terminal report",
                      rsp->mutable_reply());
        return grpc::Status::OK;
    }

    std::vector<AgentEpisodeResult> metric_agents;
    metric_agents.reserve(session->agents.size());
    for (const auto& item : session->agents) {
        const auto& agent = item.second;
        if (!agent.done_collected ||
            !IsEnvironmentTerminal(agent.task.final_termination_reason) ||
            agent.has_pending_action ||
            agent.segment_open || !agent.segment_id.empty() ||
            !agent.segment_transitions.empty()) {
            RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "Episode has an uncommitted Agent segment",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }
        AgentEpisodeResult metric;
        metric.agent_id = static_cast<uint32_t>(item.first);
        metric.episode_return = agent.episode_return;
        metric.success = agent.task.reached_goal;
        metric.termination_reason = agent.task.final_termination_reason;
        metric.transition_count = agent.episode_transition_count;
        metric.shortest_action_steps = session->task.shortest_action_steps;
        metric.unique_cell_count = static_cast<int64_t>(agent.task.visited.size());
        metric.blocked_move_count = agent.task.blocked_move_count;
        metric.attempted_move_count = agent.episode_transition_count;
        if (!std::isfinite(metric.episode_return) ||
            metric.transition_count <= 0 ||
            metric.shortest_action_steps <= 0 ||
            metric.unique_cell_count <= 0 ||
            metric.blocked_move_count < 0 ||
            metric.attempted_move_count < 0 ||
            metric.blocked_move_count > metric.attempted_move_count) {
            RejectCommand(*session,
                          rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "Episode metric source facts are invalid",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }
        if (session->current_episode_mode == PolicyMode::Training &&
            (!agent.episode_behavior_model_seen ||
             agent.episode_behavior_model_lineage_id.empty())) {
            RejectCommand(*session,
                          rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "training Episode has no behavior model facts",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }
        metric.minimum_behavior_model_step =
            agent.minimum_episode_behavior_model_step;
        metric.maximum_behavior_model_step =
            agent.maximum_episode_behavior_model_step;
        metric.behavior_model_lineage_id =
            agent.episode_behavior_model_lineage_id;
        if (agent.terminal_frame_id < 0) {
            RejectCommand(*session,
                          rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "Episode terminal frame is missing",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }
        metric.terminal_frame_id =
            static_cast<uint64_t>(agent.terminal_frame_id);
        metric.final_grid_x = agent.task.observation_grid_x;
        metric.final_grid_y = agent.task.observation_grid_y;
        metric.reward_component_sums = agent.reward_component_sums;
        if (session->current_episode_mode == PolicyMode::Training) {
            double component_total = 0.0;
            bool component_valid = !metric.reward_component_sums.empty();
            for (const auto& component : metric.reward_component_sums) {
                component_valid = component_valid &&
                    !component.first.empty() &&
                    std::isfinite(component.second);
                component_total += component.second;
            }
            const double tolerance = 1e-5 * std::max(
                1.0, std::max(std::abs(metric.episode_return),
                              std::abs(component_total)));
            if (!component_valid ||
                std::abs(metric.episode_return - component_total) > tolerance ||
                metric.minimum_behavior_model_step >
                    metric.maximum_behavior_model_step) {
                RejectCommand(*session,
                              rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                              "training Episode metric facts are inconsistent",
                              rsp->mutable_reply());
                return grpc::Status::OK;
            }
        }
        metric_agents.push_back(std::move(metric));
    }
    std::sort(metric_agents.begin(), metric_agents.end(),
              [](const AgentEpisodeResult& left,
                 const AgentEpisodeResult& right) {
                  return left.agent_id < right.agent_id;
              });
    std::vector<uint64_t> goal_frames;
    for (const auto& agent : metric_agents) {
        if (agent.success) goal_frames.push_back(agent.terminal_frame_id);
    }
    std::sort(goal_frames.begin(), goal_frames.end());
    goal_frames.erase(std::unique(goal_frames.begin(), goal_frames.end()),
                      goal_frames.end());
    for (auto& agent : metric_agents) {
        if (!agent.success) continue;
        agent.goal_rank_group = static_cast<uint32_t>(
            std::lower_bound(goal_frames.begin(), goal_frames.end(),
                             agent.terminal_frame_id) -
            goal_frames.begin() + 1);
    }
    SessionManager::Session candidate = *session;
    if (task_adapter_.Config().server.run_mode == aiserver_mode::kTraining &&
        session->current_episode_mode != PolicyMode::Training) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      "training sessions cannot commit evaluation episodes",
                      rsp->mutable_reply());
        return grpc::Status::OK;
    }

    candidate.phase = rl::session::v1::SESSION_PHASE_READY;
    candidate.behavior_policy_scope = BehaviorPolicyScope::Unspecified;
    candidate.evaluation_pinned_model_lineage_id.clear();
    candidate.evaluation_pinned_model_step = 0;
    const auto episode_outcome =
        task_adapter_.BuildEpisodeOutcome(*session, metric_agents);
    training::RegisteredMetricRecord metric_fact;
    if (session->current_episode_mode == PolicyMode::Training) {
        const auto ended_at = std::chrono::steady_clock::now();
        const int64_t observed_at_unix_ms = runtime_.NowMs();
        try {
            metric_fact = task_adapter_.BuildEpisodeMetricFact(runtime_.metric_registry_, *session, metric_agents);
            session->reward_metrics.AppendTo(metric_fact, runtime_.metric_registry_, "task.maze.reward.", observed_at_unix_ms, ended_at);
        } catch (const std::invalid_argument& error) {
            RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          std::string("episode metric producer failed: ") + error.what(),
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }
        std::string metric_payload;
        if (!metric_fact.SerializeToString(&metric_payload)) {
            RejectCommand(*session,
                          rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "Episode metric payload serialization failed",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }
        const auto append_result = runtime_.metric_events_.AppendFact(
            std::move(metric_payload), observed_at_unix_ms);
        if (!append_result.applied()) {
            RejectCommand(*session,
                          rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "Episode metric journal is already final",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }
        if (append_result.wall_clock_regressed) {
            LOG_WARN(
                "MetricEvent",
                "wall_clock_regression previous_observed_at_unix_ms=%lld "
                "observed_at_unix_ms=%lld episode=%s",
                static_cast<long long>(
                    append_result.previous_observed_at_unix_ms),
                static_cast<long long>(observed_at_unix_ms),
                session->current_episode_id.c_str());
        }
    }
    candidate.reward_metrics = {};
    *rsp->mutable_outcome() = episode_outcome;
    CommitCommand(candidate, req->command(), *req, rsp,
                  "Episode outcome and metrics committed");
    *session = std::move(candidate);
    return grpc::Status::OK;
}

grpc::Status MazeServiceImpl::AbortEpisode(
    grpc::ServerContext*,
    const maze::AbortEpisodeReq* req,
    maze::AbortEpisodeRsp* rsp) {
    std::lock_guard<std::mutex> lock(runtime_.mutex_);
    auto* session = runtime_.session_mgr_.GetSession(req->command().session_id());
    if (!session) {
        FillOpenRejected(rl::session::v1::COMMAND_ERROR_CODE_INVALID_IDENTITY,
                         "session does not exist", rsp->mutable_reply());
        return grpc::Status::OK;
    }
    const auto check = CheckCommand(*session, req->command(), *req, rsp);
    if (check == CommandCheck::Proceed || check == CommandCheck::Replayed) {
        session->last_valid_client_activity_unix_ms = runtime_.NowMs();
    }
    if (check != CommandCheck::Proceed) return grpc::Status::OK;
    if (runtime_.shutdown_started_) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      "AIServer is draining and no longer accepts AbortEpisode",
                      rsp->mutable_reply());
        return grpc::Status::OK;
    }
    const bool valid_reason =
        req->reason() == maze::MAZE_TERMINATION_REASON_CLIENT_ABORT ||
        req->reason() == maze::MAZE_TERMINATION_REASON_CHAIN_FAILURE;
    if (!valid_reason ||
        session->phase != rl::session::v1::SESSION_PHASE_EPISODE_RUNNING ||
        req->command().episode_id() != session->current_episode_id) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      "AbortEpisode reason or identity is invalid",
                      rsp->mutable_reply());
        return grpc::Status::OK;
    }

    SessionManager::Session candidate = *session;
    int64_t candidate_produced_transitions =
        runtime_.produced_unique_transitions_;
    int64_t candidate_produced_envelopes =
        runtime_.produced_unique_envelopes_;
    auto candidate_produced_by_model = runtime_.produced_transitions_by_model_;
    int64_t candidate_quarantined_transitions =
        runtime_.quarantined_transition_count_;
    int64_t candidate_pending_actions_excluded =
        runtime_.pending_action_excluded_count_;
    int64_t candidate_closed_segments = runtime_.closed_segment_count_;
    auto candidate_close_counts = runtime_.segment_close_counts_;
    std::vector<training::ProcessedTransitionEnvelope> envelopes;
    std::string error;

    const bool training_episode =
        task_adapter_.Config().server.run_mode == aiserver_mode::kTraining &&
        candidate.current_episode_mode == PolicyMode::Training;
    const auto close_reason =
        training::SEGMENT_CLOSE_REASON_CLIENT_CONTROLLED_CLOSE;
    for (auto& item : candidate.agents) {
        auto& agent = item.second;
        if (!training_episode) {
            agent.has_pending_action = false;
            agent.pending_action_frame_id = -1;
            agent.pending_obs.clear();
            agent.pending_action_mask.clear();
            continue;
        }
        if (!agent.segment_open && !agent.has_pending_action &&
            agent.segment_transitions.empty()) {
            continue;
        }
        if (agent.segment_open && !agent.segment_transitions.empty()) {
            const bool had_pending_action = agent.has_pending_action;
            float bootstrap_value = 0.0f;
            if (had_pending_action) {
                bootstrap_value = agent.pending_value;
                if (!std::isfinite(bootstrap_value)) {
                    ++runtime_.rollout_estimator_failure_count_;
                    runtime_.MarkDegraded(
                        "controlled close pending bootstrap is non-finite");
                    RejectCommand(*session,
                                  rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                                  runtime_.last_error_, rsp->mutable_reply());
                    return grpc::Status::OK;
                }
            } else if (!runtime_.InferPinnedValue(
                           agent,
                           agent.segment_transitions.back().next_observation,
                           bootstrap_value)) {
                ++runtime_.rollout_estimator_failure_count_;
                RejectCommand(*session,
                              rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                              runtime_.last_error_, rsp->mutable_reply());
                return grpc::Status::OK;
            }
            if (!runtime_.PrepareAgentSegmentClose(
                    candidate, item.first, close_reason,
                    bootstrap_value, true,
                    candidate_produced_transitions,
                    candidate_produced_envelopes,
                    candidate_produced_by_model, envelopes,
                    candidate_close_counts, error)) {
                ++runtime_.rollout_estimator_failure_count_;
                runtime_.MarkDegraded("controlled Agent segment close failed: " +
                             error);
                RejectCommand(*session,
                              rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                              runtime_.last_error_, rsp->mutable_reply());
                return grpc::Status::OK;
            }
            ++candidate_closed_segments;
            if (had_pending_action) {
                ++candidate_pending_actions_excluded;
                agent.has_pending_action = false;
                agent.pending_action_frame_id = -1;
                agent.pending_obs.clear();
                agent.pending_action_mask.clear();
            }
        } else {
            runtime_.DiscardAgentSegment(
                candidate, item.first, close_reason, true,
                candidate_quarantined_transitions,
                candidate_pending_actions_excluded,
                candidate_closed_segments, candidate_close_counts);
        }
    }

    uint64_t reservation_id = 0;
    const auto reservation =
        runtime_.sample_distributor_.ReserveEnqueueEnvelopeSet(
            envelopes, reservation_id, error);
    if (reservation ==
        SampleDistributor::ReservationResult::kRetryableUnavailable) {
        FillReply(
            *session, session->last_command_sequence,
            rl::session::v1::COMMAND_RESULT_WAIT,
            rl::session::v1::COMMAND_ERROR_CODE_UNSPECIFIED,
            error.empty()
                ? "SampleDistributor cannot reserve controlled-close output"
                : error,
            rsp->mutable_reply());
        rsp->mutable_wait()->set_retry_after_ms(
            runtime_.sample_distributor_.PauseRetryAfterMs());
        return grpc::Status::OK;
    }
    if (reservation ==
        SampleDistributor::ReservationResult::kTerminalFault) {
        runtime_.MarkDegraded("controlled-close envelope reservation failed: " +
                     error);
        RejectCommand(*session,
                      rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      runtime_.last_error_, rsp->mutable_reply());
        return grpc::Status::OK;
    }
    const auto seal =
        runtime_.sample_distributor_.SealEnqueueEnvelopeSet(reservation_id, error);
    if (seal == SampleDistributor::SealResult::kRetryableUnavailable) {
        FillReply(
            *session, session->last_command_sequence,
            rl::session::v1::COMMAND_RESULT_WAIT,
            rl::session::v1::COMMAND_ERROR_CODE_UNSPECIFIED,
            error.empty()
                ? "SampleDistributor changed before controlled-close seal"
                : error,
            rsp->mutable_reply());
        rsp->mutable_wait()->set_retry_after_ms(
            runtime_.sample_distributor_.PauseRetryAfterMs());
        return grpc::Status::OK;
    }
    if (seal == SampleDistributor::SealResult::kTerminalFault) {
        runtime_.MarkDegraded("controlled-close envelope seal failed: " + error);
        RejectCommand(*session,
                      rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      runtime_.last_error_, rsp->mutable_reply());
        return grpc::Status::OK;
    }

    candidate.phase = rl::session::v1::SESSION_PHASE_ABORTED;
    candidate.behavior_policy_scope = BehaviorPolicyScope::Unspecified;
    candidate.evaluation_pinned_model_lineage_id.clear();
    candidate.evaluation_pinned_model_step = 0;
    CommitCommand(candidate, req->command(), *req, rsp,
                  "Episode aborted after controlled Agent segment close");

    if (runtime_.sample_distributor_.CommitEnqueueEnvelopeSet(
            reservation_id, error) !=
        SampleDistributor::CommitResult::kCommitted) {
        runtime_.MarkDegraded(
            "controlled-close sealed envelope commit failed: " + error);
        RejectCommand(*session,
                      rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      runtime_.last_error_, rsp->mutable_reply());
        return grpc::Status::OK;
    }

    *session = std::move(candidate);
    runtime_.produced_unique_transitions_ = candidate_produced_transitions;
    runtime_.produced_unique_envelopes_ = candidate_produced_envelopes;
    runtime_.produced_transitions_by_model_ =
        std::move(candidate_produced_by_model);
    runtime_.quarantined_transition_count_ =
        candidate_quarantined_transitions;
    runtime_.pending_action_excluded_count_ =
        candidate_pending_actions_excluded;
    runtime_.closed_segment_count_ = candidate_closed_segments;
    runtime_.segment_close_counts_ = std::move(candidate_close_counts);
    return grpc::Status::OK;
}

grpc::Status MazeServiceImpl::CloseSession(
    grpc::ServerContext*,
    const maze::CloseSessionReq* req,
    maze::CloseSessionRsp* rsp) {
    std::lock_guard<std::mutex> lock(runtime_.mutex_);
    auto* session = runtime_.session_mgr_.GetSession(req->command().session_id());
    if (!session) {
        FillOpenRejected(rl::session::v1::COMMAND_ERROR_CODE_INVALID_IDENTITY,
                         "session does not exist", rsp->mutable_reply());
        return grpc::Status::OK;
    }
    const auto check = CheckCommand(*session, req->command(), *req, rsp);
    if (check == CommandCheck::Proceed || check == CommandCheck::Replayed) {
        session->last_valid_client_activity_unix_ms = runtime_.NowMs();
    }
    if (check != CommandCheck::Proceed) return grpc::Status::OK;
    const bool closeable =
        session->phase == rl::session::v1::SESSION_PHASE_OPEN ||
        session->phase == rl::session::v1::SESSION_PHASE_READY ||
        session->phase == rl::session::v1::SESSION_PHASE_ABORTED ||
        session->phase == rl::session::v1::SESSION_PHASE_TASK_COMPLETE;
    if (!closeable) {
        RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                      "Session close requires no active Episode",
                      rsp->mutable_reply());
        return grpc::Status::OK;
    }
    session->phase = rl::session::v1::SESSION_PHASE_CLOSED;
    CommitCommand(*session, req->command(), *req, rsp,
                  "Session closed without live resources");
    return grpc::Status::OK;
}
