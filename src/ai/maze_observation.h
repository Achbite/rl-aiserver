#pragma once

#include "session/session_manager.h"

#include <cstdint>
#include <string>
#include <vector>

class MazeObservation {
public:
    static constexpr int kDimension = 17;

    static bool Build(const SessionManager::Session& session,
                      const SessionManager::AgentRuntime& agent,
                      int gx,
                      int gy,
                      int64_t episode_step,
                      int ray_max_range,
                      int expected_obs_dim,
                      std::vector<float>& observation,
                      std::string& error);

    static bool ApplyState(const SessionManager::Session& session,
                           SessionManager::AgentRuntime& agent,
                           int gx,
                           int gy,
                           int64_t frame_id,
                           bool is_done,
                           bool reported_last_move_blocked,
                           std::string& error);
};
