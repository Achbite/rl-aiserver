#include "ai/maze_reward.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool IsTaskTerminal(maze::MazeTerminationReason reason) {
    return reason == maze::MAZE_TERMINATION_REASON_GOAL_REACHED ||
           reason == maze::MAZE_TERMINATION_REASON_TIME_LIMIT;
}

bool DistanceAt(const SessionManager::Session& session,
                int gx, int gy, int& distance) {
    if (gx < 0 || gx >= session.grid_cols ||
        gy < 0 || gy >= session.grid_rows) {
        return false;
    }
    const std::size_t index =
        static_cast<std::size_t>(gy * session.grid_cols + gx);
    if (index >= session.geodesic_distance.size()) return false;
    distance = session.geodesic_distance[index];
    return distance >= 0;
}

float FirstVisitBudget(maze::CurriculumStage stage,
                       const MazeRewardConfig& config) {
    switch (stage) {
        case maze::CURRICULUM_STAGE_8X:
            return config.stage_8x_first_visit_budget;
        case maze::CURRICULUM_STAGE_4X:
            return config.stage_4x_first_visit_budget;
        case maze::CURRICULUM_STAGE_2X:
            return config.stage_2x_first_visit_budget;
        default:
            return 0.0f;
    }
}

RewardDetail Invalid(std::string error) {
    RewardDetail detail;
    detail.valid = false;
    detail.error = std::move(error);
    return detail;
}

}  // namespace

RewardDetail MazeReward::Calculate(
    const SessionManager::Session& session,
    int agent_id, int gx, int gy, bool is_done,
    maze::MazeTerminationReason reason,
    const MazeRewardConfig& config) {
    const auto agent_it = session.agents.find(agent_id);
    if (agent_it == session.agents.end()) {
        return Invalid("reward agent identity is unknown");
    }
    const auto& agent = agent_it->second;
    if (agent.prev_grid_x < 0 || agent.prev_grid_y < 0) {
        return Invalid("reward transition has no previous state");
    }
    if (session.shortest_action_steps <= 0 ||
        agent.episode_start_geodesic_distance <= 0 ||
        !std::isfinite(config.goal_reward) ||
        !std::isfinite(config.timeout_penalty) ||
        !std::isfinite(config.progress_budget) ||
        !std::isfinite(config.stage_8x_first_visit_budget) ||
        !std::isfinite(config.stage_4x_first_visit_budget) ||
        !std::isfinite(config.stage_2x_first_visit_budget) ||
        !std::isfinite(config.wasted_action_penalty) ||
        config.goal_reward <= 0.0f ||
        config.timeout_penalty >= 0.0f ||
        config.progress_budget < 0.0f ||
        config.stage_8x_first_visit_budget < 0.0f ||
        config.stage_4x_first_visit_budget < 0.0f ||
        config.stage_2x_first_visit_budget < 0.0f ||
        config.stage_8x_first_visit_budget <
            config.stage_4x_first_visit_budget ||
        config.stage_4x_first_visit_budget <
            config.stage_2x_first_visit_budget ||
        config.stage_2x_first_visit_budget != 0.0f ||
        config.wasted_action_penalty >= 0.0f ||
        config.timeout_penalty + config.progress_budget +
                config.stage_8x_first_visit_budget >=
            0.0f) {
        return Invalid("Reward V4 configuration or episode distance is invalid");
    }
    if (is_done != IsTaskTerminal(reason)) {
        return Invalid("reward termination reason is inconsistent");
    }
    if (reason == maze::MAZE_TERMINATION_REASON_GOAL_REACHED &&
        (gx != session.end_gx || gy != session.end_gy)) {
        return Invalid("goal termination was reported outside the goal cell");
    }
    if (reason != maze::MAZE_TERMINATION_REASON_GOAL_REACHED &&
        gx == session.end_gx && gy == session.end_gy) {
        return Invalid("goal cell requires GOAL_REACHED termination");
    }
    if (reason == maze::MAZE_TERMINATION_REASON_TIME_LIMIT &&
        gx == session.end_gx && gy == session.end_gy) {
        return Invalid("goal cell cannot be reported as TIME_LIMIT");
    }

    int previous_distance = -1;
    int current_distance = -1;
    if (!DistanceAt(session, agent.prev_grid_x, agent.prev_grid_y,
                    previous_distance) ||
        !DistanceAt(session, gx, gy, current_distance)) {
        return Invalid("reward transition entered an unreachable map cell");
    }
    if (std::abs(previous_distance - current_distance) > 1) {
        return Invalid("reward transition has an illegal geodesic distance delta");
    }

    RewardDetail detail;
    const bool goal =
        reason == maze::MAZE_TERMINATION_REASON_GOAL_REACHED;
    const bool timeout =
        reason == maze::MAZE_TERMINATION_REASON_TIME_LIMIT;
    const float goal_reward = goal ? config.goal_reward : 0.0f;
    const float timeout_penalty =
        timeout ? config.timeout_penalty : 0.0f;
    const float distance_normalizer = static_cast<float>(
        agent.episode_start_geodesic_distance);
    const float geodesic_progress =
        config.progress_budget *
        static_cast<float>(previous_distance - current_distance) /
        distance_normalizer;

    const float first_visit_budget =
        FirstVisitBudget(session.curriculum_stage, config);
    const float first_visit_scale =
        first_visit_budget / distance_normalizer;
    float first_visit_bonus = 0.0f;
    const bool moved = gx != agent.prev_grid_x || gy != agent.prev_grid_y;
    if (moved && agent.current_state_first_visit &&
        first_visit_scale > 0.0f) {
        first_visit_bonus = std::min(
            first_visit_scale,
            std::max(0.0f,
                     first_visit_budget - agent.first_visit_bonus_total));
    }
    const float wasted_action_penalty =
        moved ? 0.0f : config.wasted_action_penalty;

    detail.items.emplace_back("goal_reward", goal_reward);
    detail.items.emplace_back("timeout_penalty", timeout_penalty);
    detail.items.emplace_back("geodesic_progress", geodesic_progress);
    detail.items.emplace_back("first_visit_bonus", first_visit_bonus);
    detail.items.emplace_back("wasted_action_penalty", wasted_action_penalty);
    detail.task_total = goal_reward + timeout_penalty;
    detail.shaping_total =
        geodesic_progress + first_visit_bonus + wasted_action_penalty;
    detail.total = detail.task_total + detail.shaping_total;

    if (!std::isfinite(detail.total) ||
        !std::isfinite(detail.task_total) ||
        !std::isfinite(detail.shaping_total)) {
        return Invalid("reward calculation produced a non-finite value");
    }
    for (const auto& item : detail.items) {
        if (!std::isfinite(item.second)) {
            return Invalid("reward component is non-finite: " + item.first);
        }
    }
    return detail;
}
