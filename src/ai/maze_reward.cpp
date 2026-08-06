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

float Potential(int distance, int maximum_distance) {
    return std::clamp(
        1.0f - static_cast<float>(distance) /
                   static_cast<float>(maximum_distance),
        0.0f, 1.0f);
}

float FirstVisitCap(maze::CurriculumStage stage,
                    const MazeRewardConfig& config) {
    switch (stage) {
        case maze::CURRICULUM_STAGE_8X:
            return config.stage_8x_first_visit_cap;
        case maze::CURRICULUM_STAGE_4X:
            return config.stage_4x_first_visit_cap;
        case maze::CURRICULUM_STAGE_2X:
            return 0.0f;
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
        config.potential_distance_scale < session.shortest_action_steps ||
        !std::isfinite(config.goal_reward) ||
        !std::isfinite(config.timeout_base) ||
        !std::isfinite(config.gamma) || config.gamma <= 0.0f ||
        config.gamma > 1.0f) {
        return Invalid("reward map or episode horizon is invalid");
    }
    if (is_done != IsTaskTerminal(reason)) {
        return Invalid("reward termination reason is inconsistent");
    }
    if (reason == maze::MAZE_TERMINATION_REASON_GOAL_REACHED &&
        (gx != session.end_gx || gy != session.end_gy)) {
        return Invalid("goal termination was reported outside the goal cell");
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

    RewardDetail detail;
    const bool goal =
        reason == maze::MAZE_TERMINATION_REASON_GOAL_REACHED;
    const bool timeout =
        reason == maze::MAZE_TERMINATION_REASON_TIME_LIMIT;
    const float goal_reward = goal ? config.goal_reward : 0.0f;
    const float timeout_base = timeout ? config.timeout_base : 0.0f;
    const float timeout_progress = timeout
        ? static_cast<float>(session.shortest_action_steps - current_distance) /
              static_cast<float>(session.shortest_action_steps)
        : 0.0f;

    const float previous_potential = Potential(
        previous_distance, config.potential_distance_scale);
    const float next_potential = is_done
                                     ? 0.0f
                                     : Potential(
                                           current_distance,
                                           config.potential_distance_scale);
    const float geodesic_pbrs =
        config.gamma * next_potential - previous_potential;

    const float first_visit_cap =
        FirstVisitCap(session.curriculum_stage, config);
    const float first_visit_scale = first_visit_cap /
        static_cast<float>(session.shortest_action_steps);
    float first_visit_bonus = 0.0f;
    const bool moved = gx != agent.prev_grid_x || gy != agent.prev_grid_y;
    if (moved && agent.current_state_first_visit &&
        first_visit_scale > 0.0f) {
        first_visit_bonus = std::min(
            first_visit_scale,
            std::max(0.0f,
                     first_visit_cap - agent.first_visit_bonus_total));
    }

    detail.items.emplace_back("goal_reward", goal_reward);
    detail.items.emplace_back("timeout_base", timeout_base);
    detail.items.emplace_back("timeout_progress", timeout_progress);
    detail.items.emplace_back("geodesic_pbrs", geodesic_pbrs);
    detail.items.emplace_back("first_visit_bonus", first_visit_bonus);
    detail.task_total =
        goal_reward + timeout_base + timeout_progress;
    detail.shaping_total = geodesic_pbrs + first_visit_bonus;
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
