#pragma once

#include "maze/episode/session.h"

#include <cstdint>
#include <string>
#include <vector>

class MazeObservation {
public:
    static constexpr int kDimension = 17;

    static bool Build(const MazeSessionManager::Session& session,
                      const MazeSessionManager::AgentRuntime& agent,
                      int gx,
                      int gy,
                      int64_t episode_step,
                      int ray_max_range,
                      int expected_obs_dim,
                      std::vector<float>& observation,
                      std::string& error);

    static bool ApplyState(const MazeSessionManager::Session& session,
                           MazeSessionManager::AgentRuntime& agent,
                           int gx,
                           int gy,
                           int64_t frame_id,
                           bool is_done,
                           bool reported_last_move_blocked,
                           std::string& error);
};
