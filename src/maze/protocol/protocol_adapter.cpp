#include "maze/protocol/adapter.h"
#include "maze/observation/observation.h"
#include "maze/environment/map.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <unistd.h>

namespace {
MazeModelIdentity ModelIdentity(const ModelManifest& model) {
    MazeModelIdentity identity;
    identity.model_step = model.model_step();
    identity.model_lineage_id = model.model_lineage_id();
    identity.trained_samples = model.trained_samples();
    return identity;
}

bool IsEnvironmentTerminal(maze::MazeTerminationReason reason) {
    return reason == maze::MAZE_TERMINATION_REASON_GOAL_REACHED ||
           reason == maze::MAZE_TERMINATION_REASON_TIME_LIMIT;
}
}

bool MazeTaskAdapter::WriteTaskControllerReceipt(
    const MazeEpisodeController& controller,
    std::string& error) const {
    namespace fs = std::filesystem;
    const fs::path root(config_.model.local_train_dir);
    std::error_code filesystem_error;
    fs::create_directories(root, filesystem_error);
    if (filesystem_error) {
        error = "cannot create task receipt directory: " +
                filesystem_error.message();
        return false;
    }
    const fs::path destination = root / "single-map-task-result.json";
    const fs::path temporary =
        destination.string() + ".tmp." + std::to_string(getpid());
    {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output) {
            error = "cannot open temporary task receipt";
            return false;
        }
        output << controller.ToJson() << '\n';
        output.flush();
        if (!output) {
            fs::remove(temporary, filesystem_error);
            error = "cannot flush temporary task receipt";
            return false;
        }
    }
    fs::rename(temporary, destination, filesystem_error);
    if (filesystem_error) {
        fs::remove(temporary, filesystem_error);
        error = "cannot publish task receipt: " +
                filesystem_error.message();
        return false;
    }
    return true;
}

void MazeTaskAdapter::FillOpen(Session& session, maze::OpenSessionRsp& response) const {
    session.task.map_id = config_.task.fixed_map_id;
    auto* environment = response.mutable_environment();
    environment->set_agent_count(
        static_cast<std::uint32_t>(config_.environment.agent_count));
    environment->set_map_id(config_.task.fixed_map_id);
    environment->set_episode_max_steps(
        static_cast<std::uint32_t>(config_.task.episode_max_steps));
    environment->set_action_mask_mode(
        config_.policy.action_mask_mode == "required"
            ? maze::ACTION_MASK_MODE_REQUIRED
            : maze::ACTION_MASK_MODE_DISABLED);
    response.set_workload_mode(session.workload_mode == PolicyMode::Training
        ? maze::WORKLOAD_MODE_TRAINING : maze::WORKLOAD_MODE_EVALUATION);
}

bool MazeTaskAdapter::InitializeTask(Session& candidate, State& state,
    const maze::InitReq& request, const ModelManifest& model, int64_t produced, TaskError& failure) const {
    const auto* req = &request;
    ValidatedMazeMap validated;
    std::string error;
    if (!ValidateMazeMapDescriptor(req->map(), config_.task.fixed_map_id,
                                   validated, error)) {
        return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_TASK_INPUT_INVALID, error.empty() ? "map descriptor does not match assignment"
                                    : error);
    }

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
    for (int agent_id = 0; agent_id < config_.environment.agent_count;
         ++agent_id) {
        candidate.agents[agent_id];
    }
    if (config_.server.run_mode == aiserver_mode::kTraining) {
        if (!state.Initialize(
                config_.task.episode_max_steps, ModelIdentity(model),
                produced, error) ||
            !WriteTaskControllerReceipt(state, error)) {
            return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT, "TaskController initialization failed: " + error);
        }
    }
    return true;
}

bool MazeTaskAdapter::AssignEpisode(Session& candidate, State& state, const ModelManifest& model,
    int64_t produced, const std::string& episode_id, maze::BeginEpisodeRsp& response, TaskError& failure) const {
    MazeEpisodePlan plan;
    const ModelManifest& planned_model = model;
    MazeModelIdentity planned_model_identity = ModelIdentity(model);
    std::string error;
    if (config_.server.run_mode == aiserver_mode::kTraining) {
        if (!state.PlanNextEpisode(
                planned_model_identity, produced,
                plan, error)) {
            return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                "TaskController planning failed: " + error, TaskError::Kind::Runtime);
        }
        if (plan.episode_mode != maze::EPISODE_MODE_TRAINING) {
            return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                "training TaskController attempted to schedule evaluation", TaskError::Kind::Runtime);
        }
    } else {
        plan.episode_mode = maze::EPISODE_MODE_EVALUATION;
        plan.max_steps = config_.task.episode_max_steps;
        plan.model = ModelIdentity(model);
    }

    maze::EpisodeAssignment assignment;
    const bool training_workload =
        config_.server.run_mode == aiserver_mode::kTraining;
    const bool training_identity_matches =
        !training_workload ||
        (plan.model.model_step == planned_model.model_step() &&
         plan.model.trained_samples == planned_model.trained_samples() &&
         plan.model.model_lineage_id == planned_model.model_lineage_id() &&
         !planned_model.model_lineage_id().empty());
    if (plan.max_steps <= 0 ||
        !training_identity_matches) {
        return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_MODEL_IDENTITY_MISMATCH, "Episode plan does not match the loaded model identity");
    }

    candidate.task.current_max_steps = plan.max_steps;
    ResetEpisodeState(candidate, episode_id);
    assignment.set_episode_id(episode_id);
    assignment.set_mode(plan.episode_mode);
    assignment.set_max_steps(static_cast<std::uint32_t>(plan.max_steps));
    response.mutable_assignment()->CopyFrom(assignment);
    return true;
}

bool MazeTaskAdapter::DecodeFrame(Session& candidate, const maze::UpdateReq& request, bool collect,
    std::vector<AgentTaskInput>& inputs, TaskError& failure) const {
    const auto* req = &request;
    const auto active_agent_count = static_cast<int>(std::count_if(
        candidate.agents.begin(), candidate.agents.end(),
        [](const auto& item) { return !item.second.done_collected; }));
    if (req->frame_id() !=
            static_cast<std::uint64_t>(candidate.last_frame_id + 1) ||
        req->agents_size() != active_agent_count) {
        return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_OUT_OF_ORDER, "frame is not contiguous or does not contain every "
                      "active Agent exactly once");
    }
    if (candidate.task.current_max_steps <= 0 ||
        req->frame_id() >
            static_cast<std::uint64_t>(candidate.task.current_max_steps)) {
        return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_OUT_OF_ORDER, "frame exceeds the assigned Episode horizon");
    }
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
            return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_INVALID_IDENTITY, "Agent identity or position is invalid");
        }
        auto& agent = agent_it->second;
        if (agent.done_collected) {
            return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT, "terminal Agent cannot be reported again");
        }

        const bool mask_required =
            config_.policy.action_mask_mode == "required";
        const bool mask_has_available =
            std::any_of(state.action_mask().begin(),
                        state.action_mask().end(),
                        [](bool available) { return available; });
        if ((state.is_done() && state.action_mask_size() != 0) ||
            (!state.is_done() && mask_required &&
             (state.action_mask_size() !=
                  config_.model.expected_action_dim ||
              !mask_has_available)) ||
            (!state.is_done() && !mask_required &&
             state.action_mask_size() != 0)) {
            return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT, "Agent action mask contradicts the session mode");
        }

        const bool initial_observation = req->frame_id() == 0;
        if ((initial_observation && state.has_executed_action_id()) ||
            (!initial_observation &&
             (!state.has_executed_action_id() ||
              state.executed_action_id() < 0 ||
              state.executed_action_id() > 8 ||
              !agent.has_pending_action ||
              state.executed_action_id() != agent.pending_action))) {
            return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT, initial_observation
                    ? "initial Agent state cannot report an executed action"
                    : "executed action receipt does not match the pending "
                      "AIServer action");
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
            return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT, "Agent state or termination reason is invalid");
        }
        std::string observation_error;
        if (!MazeObservation::ApplyState(
                candidate, agent, gx, gy,
                static_cast<int64_t>(req->frame_id()), state.is_done(),
                state.last_move_blocked(), observation_error)) {
            return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT, observation_error);
        }

        AgentTaskInput input;
        input.agent_id = agent_id;
        input.terminal = state.is_done();
        input.action_mask.assign(state.action_mask().begin(), state.action_mask().end());
        std::string error;
        if (agent.has_pending_action && collect) {
            auto reward = CalculateReward(candidate, agent, gx, gy, state.is_done(), state.termination_reason());
            if (!reward.valid) return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                "Episode transition preparation failed: " + reward.error, TaskError::Kind::Rollout);
            input.reward = reward;
            for (const auto& item : reward.items) {
                if (item.first == "first_visit_bonus") agent.task.first_visit_bonus_total += item.second;
            }
        }
        if ((!state.is_done() || (agent.has_pending_action && collect)) &&
            !BuildObservation(candidate, agent, gx, gy, static_cast<int64_t>(req->frame_id()), input.observation, error)) {
            return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                "maze.observation construction failed: " + error, TaskError::Kind::Rollout);
        }
        if (state.is_done()) {
            agent.task.reached_goal = goal;
            agent.task.final_termination_reason = state.termination_reason();
        }
        inputs.push_back(std::move(input));
    }
    return true;
}

void MazeTaskAdapter::EncodeActions(Session& session, const std::vector<ModelTaskAction>& actions,
    maze::UpdateRsp& response) const {
    auto* batch = response.mutable_action_batch();
    for (const auto& action : actions) {
        auto* target = batch->add_actions();
        target->set_agent_id(action.agent_id);
        target->set_action_id(static_cast<maze::MazeAction>(action.action));
        auto& agent = session.agents.at(action.agent_id);
        agent.task.prev_grid_x = agent.task.observation_grid_x;
        agent.task.prev_grid_y = agent.task.observation_grid_y;
    }
}

bool MazeTaskAdapter::ObserveProgress(State& state, const ModelManifest& model,
    int64_t produced, TaskError& failure) const {
    std::string error;
    if (!state.ObserveTrainingProgress(ModelIdentity(model), produced, error))
        return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
            "TaskController collection check failed: " + error, TaskError::Kind::Runtime);
    return true;
}

bool MazeTaskAdapter::PrepareOutcome(const Session& session, maze::EndEpisodeRsp& response,
    MetricRegistry& registry, training::RegisteredMetricRecord& fact, TaskError& failure) const {
    std::vector<AgentEpisodeResult> metric_agents;
    metric_agents.reserve(session.agents.size());
    for (const auto& item : session.agents) {
        const auto& agent = item.second;
        if (!IsEnvironmentTerminal(agent.task.final_termination_reason)) {
            return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                "Episode termination reason is invalid");
        }
        AgentEpisodeResult metric;
        metric.agent_id = static_cast<uint32_t>(item.first);
        metric.episode_return = agent.episode_return;
        metric.success = agent.task.reached_goal;
        metric.termination_reason = agent.task.final_termination_reason;
        metric.transition_count = agent.episode_transition_count;
        metric.shortest_action_steps = session.task.shortest_action_steps;
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
            return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT, "Episode metric source facts are invalid");
        }
        if (session.current_episode_mode == PolicyMode::Training &&
            (!agent.episode_behavior_model_seen ||
             agent.episode_behavior_model_lineage_id.empty())) {
            return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT, "training Episode has no behavior model facts");
        }
        metric.minimum_behavior_model_step =
            agent.minimum_episode_behavior_model_step;
        metric.maximum_behavior_model_step =
            agent.maximum_episode_behavior_model_step;
        metric.behavior_model_lineage_id =
            agent.episode_behavior_model_lineage_id;
        if (agent.terminal_frame_id < 0) {
            return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT, "Episode terminal frame is missing");
        }
        metric.terminal_frame_id =
            static_cast<uint64_t>(agent.terminal_frame_id);
        metric.final_grid_x = agent.task.observation_grid_x;
        metric.final_grid_y = agent.task.observation_grid_y;
        metric.reward_component_sums = agent.reward_component_sums;
        if (session.current_episode_mode == PolicyMode::Training) {
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
                return failure.Set(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT, "training Episode metric facts are inconsistent");
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
    *response.mutable_outcome() = BuildEpisodeOutcome(session, metric_agents);
    if (session.current_episode_mode == PolicyMode::Training)
        fact = BuildEpisodeMetricFact(registry, session, metric_agents);
    return true;
}

bool MazeTaskAdapter::ValidAbort(const maze::AbortEpisodeReq& request) const {
    return request.reason() == maze::MAZE_TERMINATION_REASON_CLIENT_ABORT ||
           request.reason() == maze::MAZE_TERMINATION_REASON_CHAIN_FAILURE;
}
