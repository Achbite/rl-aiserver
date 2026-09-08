#pragma once

#include "contracts/contract_namespaces.h"
#include "metrics/metric_registry.h"
#include <algorithm>
#include <optional>
#include <unordered_map>
#include <vector>
#include "proto/metrics/registry.pb.h"

struct AgentEpisodeResult {
    uint32_t agent_id = 0;
    double episode_return = 0.0;
    bool success = false;
    maze::MazeTerminationReason termination_reason =
        maze::MAZE_TERMINATION_REASON_UNSPECIFIED;
    int64_t transition_count = 0;
    int64_t shortest_action_steps = 0;
    int64_t unique_cell_count = 0;
    int64_t blocked_move_count = 0;
    int64_t attempted_move_count = 0;
    uint64_t minimum_behavior_model_step = 0;
    uint64_t maximum_behavior_model_step = 0;
    std::string behavior_model_lineage_id;
    uint64_t terminal_frame_id = 0;
    int final_grid_x = 0;
    int final_grid_y = 0;
    std::optional<uint32_t> goal_rank_group;
    std::unordered_map<std::string, double> reward_component_sums;
};


inline void RegisterMazeEpisodeMetrics(MetricRegistry& registry) {
    auto mean = [&](const std::string& id, const std::string& label, const std::string& unit,
                    const std::string& scope, const std::string& denominator, const std::string& category) {
        registry.Register(id, label, unit, scope, training::METRIC_VALUE_TYPE_SUM_COUNT,
                          training::METRIC_AGGREGATION_MEAN, denominator, category);
    };
    auto count = [&](const std::string& id, const std::string& label, const std::string& scope) {
        registry.Register(id, label, "count", scope, training::METRIC_VALUE_TYPE_UNSIGNED,
                          training::METRIC_AGGREGATION_SUM, {}, "episode");
    };
    count("task.maze.environment_episodes", "Environment Episodes", "environment_episode");
    count("task.maze.agent_episodes", "Agent Episodes", "agent_episode");
    mean("task.maze.any_success", "Any Agent Success Rate", "ratio", "environment_episode", "environment_episode", "success");
    mean("task.maze.all_success", "All Agents Success Rate", "ratio", "environment_episode", "environment_episode", "success");
    mean("task.maze.return", "Mean Episode Return", "reward", "agent_episode", "agent_episode", "reward");
    mean("task.maze.success", "Agent Success Rate", "ratio", "agent_episode", "agent_episode", "success");
    mean("task.maze.episode_length", "Episode Length", "transition", "agent_episode", "agent_episode", "episode");
    mean("task.maze.unique_cells", "Unique Cells", "cell", "agent_episode", "agent_episode", "episode");
    mean("task.maze.blocked_move_rate", "Blocked Move Rate", "ratio", "agent_episode", "attempted_move", "episode");
    mean("task.maze.path_ratio", "Successful Path Ratio", "ratio", "agent_episode", "successful_agent_episode", "episode");
    for (const auto& item : std::vector<std::pair<std::string, training::MetricAggregation>>{
            {"min", training::METRIC_AGGREGATION_MIN}, {"max", training::METRIC_AGGREGATION_MAX}})
        registry.Register("task.maze.return_" + item.first,
            item.first == "min" ? "Episode Return Min" : "Episode Return Max", "reward", "agent_episode",
            training::METRIC_VALUE_TYPE_SCALAR, item.second, {}, "episode");
    for (int value = maze::MazeTerminationReason_MIN; value <= maze::MazeTerminationReason_MAX; ++value) {
        if (!maze::MazeTerminationReason_IsValid(value)) continue;
        const auto name = maze::MazeTerminationReason_Name(static_cast<maze::MazeTerminationReason>(value));
        count("task.maze.termination." + name, name, "agent_episode");
    }
}

inline training::RegisteredMetricRecord BuildMazeEpisodeMetrics(
    MetricRegistry& registry, const std::string& environment_id,
    const std::string& episode_id, const std::vector<AgentEpisodeResult>& agents) {
    if (environment_id.empty() || episode_id.empty() || agents.empty()) {
        throw std::invalid_argument("episode metric context is incomplete");
    }
    training::RegisteredMetricRecord record;
    (*record.mutable_attributes())["environment_instance_id"] = environment_id;
    (*record.mutable_attributes())["episode_id"] = episode_id;
    auto mean = [&](const std::string& id, const std::string& label, const std::string& unit,
                    const std::string& scope, const std::string& denominator,
                    double sum, uint64_t count) {
        registry.Mean(record, id, sum, count);
    };
    auto count = [&](const std::string& id, const std::string& label, const std::string& scope, uint64_t value) {
        registry.Unsigned(record, id, value);
    };
    count("task.maze.environment_episodes", "Environment Episodes", "environment_episode", 1);
    const auto successes = std::count_if(agents.begin(), agents.end(),
        [](const auto& agent) { return agent.success; });
    mean("task.maze.any_success", "Any Agent Success Rate", "ratio", "environment_episode", "environment_episode",
         successes > 0 ? 1.0 : 0.0, 1);
    mean("task.maze.all_success", "All Agents Success Rate", "ratio", "environment_episode", "environment_episode",
         successes == static_cast<int64_t>(agents.size()) ? 1.0 : 0.0, 1);
    for (const auto& agent : agents) {
        const int first_point = record.points_size();
        if (agent.transition_count < 0 || agent.blocked_move_count < 0 ||
            agent.blocked_move_count > agent.attempted_move_count ||
            agent.minimum_behavior_model_step > agent.maximum_behavior_model_step ||
            agent.behavior_model_lineage_id.empty() ||
            agent.termination_reason == maze::MAZE_TERMINATION_REASON_UNSPECIFIED) {
            throw std::invalid_argument("agent episode metric facts are contradictory");
        }
        double reward_sum = 0.0;
        for (const auto& component : agent.reward_component_sums) reward_sum += component.second;
        if (!std::isfinite(reward_sum) ||
            std::abs(reward_sum - agent.episode_return) >
                std::max(1e-6, std::abs(agent.episode_return) * 1e-6)) {
            throw std::invalid_argument("reward components do not conserve episode return");
        }
        count("task.maze.agent_episodes", "Agent Episodes", "agent_episode", 1);
        mean("task.maze.return", "Mean Episode Return", "reward", "agent_episode", "agent_episode", agent.episode_return, 1);
        mean("task.maze.success", "Agent Success Rate", "ratio", "agent_episode", "agent_episode", agent.success ? 1.0 : 0.0, 1);
        mean("task.maze.episode_length", "Episode Length", "transition", "agent_episode", "agent_episode", agent.transition_count, 1);
        mean("task.maze.unique_cells", "Unique Cells", "cell", "agent_episode", "agent_episode", agent.unique_cell_count, 1);
        if (agent.attempted_move_count > 0) {
            mean("task.maze.blocked_move_rate", "Blocked Move Rate", "ratio", "agent_episode", "attempted_move",
                 agent.blocked_move_count, agent.attempted_move_count);
        }
        if (agent.success && agent.shortest_action_steps > 0) {
            mean("task.maze.path_ratio", "Successful Path Ratio", "ratio", "agent_episode", "successful_agent_episode",
                 static_cast<double>(agent.transition_count) / agent.shortest_action_steps, 1);
        }
        for (const auto& item : std::vector<std::pair<std::string, training::MetricAggregation>>{
                {"min", training::METRIC_AGGREGATION_MIN}, {"max", training::METRIC_AGGREGATION_MAX}}) {
            const auto id = "task.maze.return_" + item.first;
            registry.Scalar(record, id, agent.episode_return);
        }
        count("task.maze.termination." + maze::MazeTerminationReason_Name(agent.termination_reason),
              maze::MazeTerminationReason_Name(agent.termination_reason), "agent_episode", 1);
        std::vector<std::pair<std::string, double>> components(
            agent.reward_component_sums.begin(), agent.reward_component_sums.end());
        std::sort(components.begin(), components.end());
        for (const auto& component : components) {
            auto label = component.first;
            std::replace(label.begin(), label.end(), '_', ' ');
            mean("task.maze.reward." + component.first + ".per_episode", label + " / Agent Episode", "reward",
                 "agent_episode", "agent_episode", component.second, 1);

        }
        for (int i = first_point; i < record.points_size(); ++i) {
            auto& attributes = *record.mutable_points(i)->mutable_attributes();
            attributes["agent_id"] = std::to_string(agent.agent_id);
            attributes["behavior_model_lineage_id"] = agent.behavior_model_lineage_id;
            attributes["minimum_behavior_model_step"] = std::to_string(agent.minimum_behavior_model_step);
            attributes["maximum_behavior_model_step"] = std::to_string(agent.maximum_behavior_model_step);
            attributes["terminal_frame_id"] = std::to_string(agent.terminal_frame_id);
        }
    }
    return record;
}
