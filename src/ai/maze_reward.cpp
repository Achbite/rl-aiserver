#include "ai/maze_reward.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace {

constexpr MazeRewardParameters kReward{
    10.0,
    -2.0,
    1.0,
    0.75,
    0.25,
    0.0,
    -0.002,
};

static_assert(kReward.goal_reward > 0.0,
              "Reward Goal reward must be positive");
static_assert(kReward.timeout_penalty < 0.0,
              "Reward timeout penalty must be negative");
static_assert(kReward.progress_budget >= 0.0 &&
                  kReward.stage_8x_first_visit_budget >=
                      kReward.stage_4x_first_visit_budget &&
                  kReward.stage_4x_first_visit_budget >=
                      kReward.stage_2x_first_visit_budget &&
                  kReward.stage_2x_first_visit_budget == 0.0,
              "Reward shaping budgets are invalid");
static_assert(kReward.wasted_action_penalty < 0.0,
              "Reward wasted-action penalty must be negative");
static_assert(kReward.timeout_penalty + kReward.progress_budget +
                      kReward.stage_8x_first_visit_budget <
                  0.0,
              "Reward maximum failure budget must remain negative");

std::string CanonicalDecimal(double value) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6) << value;
    std::string result = output.str();
    while (result.size() > 2 && result.back() == '0' &&
           result[result.size() - 2] != '.') {
        result.pop_back();
    }
    return result;
}

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

RewardDetail Invalid(std::string error) {
    RewardDetail detail;
    detail.valid = false;
    detail.error = std::move(error);
    return detail;
}

}  // namespace

const MazeRewardParameters& GetMazeRewardParameters() {
    return kReward;
}

std::string MazeRewardCanonicalParametersJson() {
    std::ostringstream output;
    output << "{\"goal_reward\":" << CanonicalDecimal(kReward.goal_reward)
           << ",\"progress_budget\":"
           << CanonicalDecimal(kReward.progress_budget)
           << ",\"stage_2x_first_visit_budget\":"
           << CanonicalDecimal(kReward.stage_2x_first_visit_budget)
           << ",\"stage_4x_first_visit_budget\":"
           << CanonicalDecimal(kReward.stage_4x_first_visit_budget)
           << ",\"stage_8x_first_visit_budget\":"
           << CanonicalDecimal(kReward.stage_8x_first_visit_budget)
           << ",\"timeout_penalty\":"
           << CanonicalDecimal(kReward.timeout_penalty)
           << ",\"wasted_action_penalty\":"
           << CanonicalDecimal(kReward.wasted_action_penalty) << '}';
    return output.str();
}

RewardDetail MazeReward::Calculate(
    const SessionManager::Session& session,
    int agent_id, int gx, int gy, bool is_done,
    maze::MazeTerminationReason reason) {
    const auto& config = GetMazeRewardParameters();
    const auto agent_it = session.agents.find(agent_id);
    if (agent_it == session.agents.end()) {
        return Invalid("reward agent identity is unknown");
    }
    const auto& agent = agent_it->second;
    if (agent.prev_grid_x < 0 || agent.prev_grid_y < 0) {
        return Invalid("reward transition has no previous state");
    }
    if (session.shortest_action_steps <= 0 ||
        agent.episode_start_geodesic_distance <= 0) {
        return Invalid("Reward episode distance is invalid");
    }
    if (is_done != IsTaskTerminal(reason)) {
        return Invalid("reward termination reason is inconsistent");
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
    const float goal_reward =
        goal ? static_cast<float>(config.goal_reward) : 0.0f;
    const float timeout_penalty =
        timeout ? static_cast<float>(config.timeout_penalty) : 0.0f;
    const float distance_normalizer = static_cast<float>(
        agent.episode_start_geodesic_distance);
    const float geodesic_progress =
        static_cast<float>(config.progress_budget) *
        static_cast<float>(previous_distance - current_distance) /
        distance_normalizer;

    // The reward contract fixes first-visit shaping to this budget. Changing it
    // or adding curriculum behavior requires a new reward contract identity.
    const float first_visit_budget =
        static_cast<float>(config.stage_8x_first_visit_budget);
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
        moved ? 0.0f : static_cast<float>(config.wasted_action_penalty);

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
