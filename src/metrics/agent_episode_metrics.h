#pragma once

#include "metrics/metric_registry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct AgentEpisodeSummary {
    uint32_t agent_id = 0;
    double episode_return = 0.0;
    int64_t transition_count = 0;
    uint64_t minimum_behavior_model_step = 0;
    uint64_t maximum_behavior_model_step = 0;
    std::string behavior_model_lineage_id;
    uint64_t terminal_frame_id = 0;
    std::unordered_map<std::string, double> reward_component_sums;
};

// The task owns field registration; episode statistics do not select task IDs.
struct AgentEpisodeMetricIds {
    std::string agent_episodes;
    std::string episode_return;
    std::string episode_length;
    std::string minimum_episode_return;
    std::string maximum_episode_return;
    std::string reward_component_prefix;
};

inline void RegisterAgentEpisodeMetrics(
    MetricRegistry& registry, const AgentEpisodeMetricIds& ids) {
    registry.Register(ids.agent_episodes, "Agent Episodes", "count", "agent_episode",
                      training::METRIC_VALUE_TYPE_UNSIGNED,
                      training::METRIC_AGGREGATION_SUM, {}, "episode");
    registry.Register(ids.episode_return, "Mean Episode Return", "reward", "agent_episode",
                      training::METRIC_VALUE_TYPE_SUM_COUNT,
                      training::METRIC_AGGREGATION_MEAN, "agent_episode", "reward");
    registry.Register(ids.episode_length, "Episode Length", "transition", "agent_episode",
                      training::METRIC_VALUE_TYPE_SUM_COUNT,
                      training::METRIC_AGGREGATION_MEAN, "agent_episode", "episode");
    registry.Register(ids.minimum_episode_return, "Episode Return Min", "reward", "agent_episode",
                      training::METRIC_VALUE_TYPE_SCALAR,
                      training::METRIC_AGGREGATION_MIN, {}, "episode");
    registry.Register(ids.maximum_episode_return, "Episode Return Max", "reward", "agent_episode",
                      training::METRIC_VALUE_TYPE_SCALAR,
                      training::METRIC_AGGREGATION_MAX, {}, "episode");
}

inline void AnnotateAgentEpisodeMetricPoints(
    training::RegisteredMetricRecord& record, int first_point,
    const AgentEpisodeSummary& agent) {
    for (int index = first_point; index < record.points_size(); ++index) {
        auto& attributes = *record.mutable_points(index)->mutable_attributes();
        attributes["agent_id"] = std::to_string(agent.agent_id);
        attributes["behavior_model_lineage_id"] = agent.behavior_model_lineage_id;
        attributes["minimum_behavior_model_step"] = std::to_string(agent.minimum_behavior_model_step);
        attributes["maximum_behavior_model_step"] = std::to_string(agent.maximum_behavior_model_step);
        attributes["terminal_frame_id"] = std::to_string(agent.terminal_frame_id);
    }
}

// Called for completed agent facts in the owner's complete Episode record.
inline void AppendAgentEpisodeMetrics(
    const MetricRegistry& registry, training::RegisteredMetricRecord& record,
    const AgentEpisodeMetricIds& ids, const AgentEpisodeSummary& agent) {
    if (agent.transition_count < 0 ||
        agent.minimum_behavior_model_step > agent.maximum_behavior_model_step ||
        agent.behavior_model_lineage_id.empty()) {
        throw std::invalid_argument("agent episode metric facts are contradictory");
    }
    double reward_sum = 0.0;
    for (const auto& component : agent.reward_component_sums) reward_sum += component.second;
    if (!std::isfinite(reward_sum) ||
        std::abs(reward_sum - agent.episode_return) >
            std::max(1e-6, std::abs(agent.episode_return) * 1e-6)) {
        throw std::invalid_argument("reward components do not conserve episode return");
    }

    const int first_point = record.points_size();
    registry.Unsigned(record, ids.agent_episodes, 1);
    registry.Mean(record, ids.episode_return, agent.episode_return, 1);
    registry.Mean(record, ids.episode_length, agent.transition_count, 1);
    registry.Scalar(record, ids.minimum_episode_return, agent.episode_return);
    registry.Scalar(record, ids.maximum_episode_return, agent.episode_return);

    std::vector<std::pair<std::string, double>> components(
        agent.reward_component_sums.begin(), agent.reward_component_sums.end());
    std::sort(components.begin(), components.end());
    for (const auto& component : components) {
        registry.Mean(record, ids.reward_component_prefix + component.first + ".per_episode",
                      component.second, 1);
    }
    AnnotateAgentEpisodeMetricPoints(record, first_point, agent);
}
