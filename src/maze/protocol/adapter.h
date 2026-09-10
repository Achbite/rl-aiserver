#pragma once

#include "maze/reward/reward.h"
#include "maze/config/config.h"
#include "maze/metrics/episode_metrics.h"
#include "maze/episode/session.h"
#include "maze/episode/episode_controller.h"
#include "task/runtime/task_input.h"
#include "task/model/model_manifest.h"

#include <cstdint>
#include <string>
#include <vector>

class MazeTaskAdapter {
public:
    using ConfigType = MazeConfig;
    using Sessions = MazeSessionManager;
    using Session = Sessions::Session;
    using State = MazeEpisodeController;
    static std::string EpisodeId(uint64_t sequence) { return "maze-episode-" + std::to_string(sequence); }
    void FillOpen(Session&, maze::OpenSessionRsp&) const;
    bool InitializeTask(Session&, State&, const maze::InitReq&, const ModelManifest&, int64_t, TaskError&) const;
    bool AssignEpisode(Session&, State&, const ModelManifest&, int64_t, const std::string&, maze::BeginEpisodeRsp&, TaskError&) const;
    bool DecodeFrame(Session&, const maze::UpdateReq&, bool, std::vector<AgentTaskInput>&, TaskError&) const;
    void EncodeActions(Session&, const std::vector<ModelTaskAction>&, maze::UpdateRsp&) const;
    bool ObserveProgress(State&, const ModelManifest&, int64_t, TaskError&) const;
    bool PrepareOutcome(const Session&, maze::EndEpisodeRsp&, MetricRegistry&, training::RegisteredMetricRecord&, TaskError&) const;
    bool ValidAbort(const maze::AbortEpisodeReq&) const;
    explicit MazeTaskAdapter(const MazeConfig& config);

    const MazeConfig& Config() const { return config_; }
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
    bool WriteTaskControllerReceipt(const MazeEpisodeController&, std::string&) const;
    const MazeConfig config_;
};
