#pragma once

#include "contracts/contract_namespaces.h"
#include "metrics/agent_episode_metrics.h"
#include <algorithm>
#include <optional>
#include <vector>
#include "proto/metrics/registry.pb.h"

struct AgentEpisodeResult : AgentEpisodeSummary {
    bool success = false;
    maze::MazeTerminationReason termination_reason =
        maze::MAZE_TERMINATION_REASON_UNSPECIFIED;
    int64_t shortest_action_steps = 0;
    int64_t unique_cell_count = 0;
    int64_t blocked_move_count = 0;
    int64_t attempted_move_count = 0;
    int final_grid_x = 0;
    int final_grid_y = 0;
    std::optional<uint32_t> goal_rank_group;
};

inline const AgentEpisodeMetricIds& MazeAgentEpisodeMetricIds() {
    static const AgentEpisodeMetricIds ids{
        "task.maze.agent_episodes",
        "task.maze.return",
        "task.maze.episode_length",
        "task.maze.return_min",
        "task.maze.return_max",
        "task.maze.reward.",
    };
    return ids;
}

inline void RegisterMazeEpisodeMetrics(MetricRegistry& registry) {
    RegisterAgentEpisodeMetrics(registry, MazeAgentEpisodeMetricIds());
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
    mean("task.maze.any_success", "Any Agent Success Rate", "ratio", "environment_episode", "environment_episode", "success");
    mean("task.maze.all_success", "All Agents Success Rate", "ratio", "environment_episode", "environment_episode", "success");
    mean("task.maze.success", "Agent Success Rate", "ratio", "agent_episode", "agent_episode", "success");
    mean("task.maze.unique_cells", "Unique Cells", "cell", "agent_episode", "agent_episode", "episode");
    mean("task.maze.blocked_move_rate", "Blocked Move Rate", "ratio", "agent_episode", "attempted_move", "episode");
    mean("task.maze.path_ratio", "Successful Path Ratio", "ratio", "agent_episode", "successful_agent_episode", "episode");
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
    registry.Unsigned(record, "task.maze.environment_episodes", 1);
    const auto successes = std::count_if(agents.begin(), agents.end(),
        [](const auto& agent) { return agent.success; });
    registry.Mean(record, "task.maze.any_success", successes > 0 ? 1.0 : 0.0, 1);
    registry.Mean(record, "task.maze.all_success",
                  successes == static_cast<int64_t>(agents.size()) ? 1.0 : 0.0, 1);
    for (const auto& agent : agents) {
        if (agent.blocked_move_count < 0 ||
            agent.blocked_move_count > agent.attempted_move_count ||
            agent.termination_reason == maze::MAZE_TERMINATION_REASON_UNSPECIFIED) {
            throw std::invalid_argument("agent episode metric facts are contradictory");
        }
        AppendAgentEpisodeMetrics(registry, record, MazeAgentEpisodeMetricIds(), agent);
        const int first_point = record.points_size();
        registry.Mean(record, "task.maze.success", agent.success ? 1.0 : 0.0, 1);
        registry.Mean(record, "task.maze.unique_cells", agent.unique_cell_count, 1);
        if (agent.attempted_move_count > 0) {
            registry.Mean(record, "task.maze.blocked_move_rate",
                          agent.blocked_move_count, agent.attempted_move_count);
        }
        if (agent.success && agent.shortest_action_steps > 0) {
            registry.Mean(record, "task.maze.path_ratio",
                          static_cast<double>(agent.transition_count) / agent.shortest_action_steps, 1);
        }
        registry.Unsigned(record,
                          "task.maze.termination." + maze::MazeTerminationReason_Name(agent.termination_reason), 1);
        AnnotateAgentEpisodeMetricPoints(record, first_point, agent);
    }
    return record;
}
