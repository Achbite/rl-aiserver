#pragma once

#include "ai/maze_reward.h"
#include "session/session_manager.h"

// Materialises one trusted Environment result. This function never invents a
// transition for a pending action whose execution result is unknown.
bool BuildRawRolloutTransition(
    const SessionManager::AgentRuntime& agent,
    const std::vector<float>& next_observation,
    const RewardDetail& reward,
    int expected_obs_dim,
    int expected_action_dim,
    const std::string& action_mask_mode,
    RawRolloutTransition& transition,
    std::string& error);
