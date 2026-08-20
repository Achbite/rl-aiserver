#pragma once

#include "ai/maze_reward.h"
#include "session/session_manager.h"

#include <string>
#include <vector>

// Materialises one trusted Environment result. This function never invents a
// transition for a pending action whose execution result is unknown.
bool BuildRawRolloutTransition(
    const SessionManager::AgentRuntime& agent,
    const std::vector<float>& next_observation,
    const RewardDetail& reward,
    int expected_obs_dim,
    int expected_action_dim,
    SessionManager::RawRolloutTransition& transition,
    std::string& error);

// Computes unnormalised backward GAE and value targets over one contiguous,
// single-Agent, single-pinned-model segment. The caller owns segment identity,
// close reason and terminal/bootstrap provenance.
bool EstimateRolloutSegment(
    const std::vector<SessionManager::RawRolloutTransition>& segment,
    double gamma,
    double gae_lambda,
    double final_next_value,
    std::vector<float>& advantages,
    std::vector<float>& value_targets,
    std::string& error);
