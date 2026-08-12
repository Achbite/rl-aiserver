#pragma once

#include "ai/maze_reward.h"
#include "contracts/contract_namespaces.h"
#include "session/session_manager.h"

#include <string>
#include <vector>

// Builds the exact wire sample consumed by the Learner from an AIServer
// pending action. The function is deliberately pure: callers may validate a
// complete multi-Agent Update before committing any Session state.
bool BuildTrainingSample(
    const SessionManager::AgentRuntime& agent,
    const std::vector<float>& next_observation,
    const RewardDetail& reward,
    bool is_done,
    maze::MazeTerminationReason reason,
    int expected_obs_dim,
    int expected_action_dim,
    const std::vector<training::Sample>& fragment,
    training::Sample& sample,
    std::string& error);
