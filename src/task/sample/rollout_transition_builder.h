#pragma once

#include "task/reward/reward_result.h"
#include "task/session/training_session.h"

// Materialises one trusted Environment result. This function never invents a
// transition for a pending action whose execution result is unknown.
bool BuildRawRolloutTransition(
    const AgentTrainingState& agent,
    const std::vector<float>& next_observation,
    const RewardResult& reward,
    int expected_obs_dim,
    int expected_action_dim,
    const std::string& action_mask_mode,
    RawRolloutTransition& transition,
    std::string& error);
