#include "ai/maze_reward.h"

#include <algorithm>
#include <cmath>
#include "proto/metrics/registry.pb.h"

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

bool IsTaskTerminal(maze::MazeTerminationReason reason) {
    return reason == maze::MAZE_TERMINATION_REASON_GOAL_REACHED ||
           reason == maze::MAZE_TERMINATION_REASON_TIME_LIMIT;
}

bool DistanceAt(const MazeRewardContext& context,
                int gx, int gy, int& distance) {
    if (gx < 0 || gx >= context.grid_cols ||
        gy < 0 || gy >= context.grid_rows) {
        return false;
    }
    const std::size_t index =
        static_cast<std::size_t>(gy * context.grid_cols + gx);
    if (index >= context.geodesic_distance.size()) return false;
    distance = context.geodesic_distance[index];
    return distance >= 0;
}

MazeRewardDetail Invalid(std::string error) {
    MazeRewardDetail detail;
    detail.valid = false;
    detail.error = std::move(error);
    return detail;
}

}  // namespace

const MazeRewardParameters& GetMazeRewardParameters() {
    return kReward;
}

MazeRewardDetail MazeReward::Calculate(
    const MazeRewardContext& context,
    int gx, int gy, bool is_done,
    maze::MazeTerminationReason reason) {
    const auto& config = GetMazeRewardParameters();
    if (context.prev_grid_x < 0 || context.prev_grid_y < 0) {
        return Invalid("reward transition has no previous state");
    }
    if (context.shortest_action_steps <= 0 ||
        context.episode_start_geodesic_distance <= 0) {
        return Invalid("Reward episode distance is invalid");
    }
    if (is_done != IsTaskTerminal(reason)) {
        return Invalid("reward termination reason is inconsistent");
    }
    int previous_distance = -1;
    int current_distance = -1;
    if (!DistanceAt(context, context.prev_grid_x, context.prev_grid_y,
                    previous_distance) ||
        !DistanceAt(context, gx, gy, current_distance)) {
        return Invalid("reward transition entered an unreachable map cell");
    }
    MazeRewardDetail detail;
    const bool goal =
        reason == maze::MAZE_TERMINATION_REASON_GOAL_REACHED;
    const bool timeout =
        reason == maze::MAZE_TERMINATION_REASON_TIME_LIMIT;
    const float goal_reward =
        goal ? static_cast<float>(config.goal_reward) : 0.0f;
    const float timeout_penalty =
        timeout ? static_cast<float>(config.timeout_penalty) : 0.0f;
    const float distance_normalizer = static_cast<float>(
        context.episode_start_geodesic_distance);
    const float geodesic_progress =
        static_cast<float>(config.progress_budget) *
        static_cast<float>(previous_distance - current_distance) /
        distance_normalizer;

    // The AIServer reward implementation owns this first-visit budget.
    const float first_visit_budget =
        static_cast<float>(config.stage_8x_first_visit_budget);
    const float first_visit_scale =
        first_visit_budget / distance_normalizer;
    float first_visit_bonus = 0.0f;
    const bool moved = gx != context.prev_grid_x || gy != context.prev_grid_y;
    if (moved && context.current_state_first_visit &&
        first_visit_scale > 0.0f) {
        first_visit_bonus = std::min(
            first_visit_scale,
            std::max(0.0f,
                     first_visit_budget - context.first_visit_bonus_total));
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

void MazeReward::RegisterMetrics(MetricRegistry& registry) {
    registry.Register("task.maze.reward.total.per_transition", "Total Reward / Transition", "reward", "agent_episode",
        training::METRIC_VALUE_TYPE_SUM_COUNT, training::METRIC_AGGREGATION_MEAN, "transition", "reward");
    for (const auto& item : std::vector<std::pair<std::string, std::string>>{
        {"goal_reward", "Goal Reward"}, {"timeout_penalty", "Timeout Penalty"},
        {"geodesic_progress", "Geodesic Progress"}, {"first_visit_bonus", "First Visit Bonus"},
        {"wasted_action_penalty", "Wasted Action Penalty"}}) {
        registry.Register("task.maze.reward." + item.first + ".per_transition", item.second + " / Transition",
            "reward", "agent_episode", training::METRIC_VALUE_TYPE_SUM_COUNT,
            training::METRIC_AGGREGATION_MEAN, "transition", "reward");
        registry.Register("task.maze.reward." + item.first + ".per_episode", item.second + " / Agent Episode",
            "reward", "agent_episode", training::METRIC_VALUE_TYPE_SUM_COUNT,
            training::METRIC_AGGREGATION_MEAN, "agent_episode", "reward");
    }
}
