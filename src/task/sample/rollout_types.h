#pragma once

#include <cstdint>
#include <vector>

// One complete, trusted Environment transition before the AIServer closes
// the Agent segment and computes GAE/value targets. Pending actions are not
// represented here because they have no trusted result or next state yet.
struct RawRolloutTransition {
    std::vector<float> observation;
    std::vector<float> next_observation;
    int action = 0;
    float reward = 0.0f;
    float behavior_log_probability = 0.0f;
    float behavior_value = 0.0f;
    std::vector<bool> action_mask;
    uint64_t action_step = 0;
    int64_t created_at_unix_ms = 0;
};
