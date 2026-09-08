#pragma once

#include "ai/maze_reward.h"
#include "task/maze_config.h"
#include "task/maze_episode_metrics.h"
#include "task/maze_session.h"

#include <cstdint>
#include <string>
#include <vector>

class MazeTaskAdapter {
public:
    explicit MazeTaskAdapter(const AIServerConfig& config);

    const AIServerConfig& Config() const { return config_; }
    void RegisterMetrics(MetricRegistry& registry) const;
    static constexpr const char* RewardMetricPrefix() { return "task.maze.reward."; }

    void ResetEpisodeState(MazeSessionManager::Session& session,
                           const std::string& episode_id) const;
    training::RegisteredMetricRecord BuildEpisodeMetricFact(
        MetricRegistry& registry, const MazeSessionManager::Session& session,
        const std::vector<AgentEpisodeResult>& agents) const;
    maze::EpisodeOutcome BuildEpisodeOutcome(
        const MazeSessionManager::Session& session,
        const std::vector<AgentEpisodeResult>& agents) const;
    bool BuildObservation(
        const MazeSessionManager::Session& session,
        const MazeSessionManager::AgentRuntime& agent,
        int gx, int gy, int64_t frame_id,
        std::vector<float>& observation, std::string& error) const;
    MazeRewardDetail CalculateReward(
        const MazeSessionManager::Session& session,
        const MazeSessionManager::AgentRuntime& agent,
        int gx, int gy, bool is_done,
        maze::MazeTerminationReason reason) const;

private:
    const AIServerConfig config_;
};
