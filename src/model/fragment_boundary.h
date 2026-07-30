#pragma once

#include <vector>

struct FragmentBoundaryState {
    bool has_pending_action = false;
    bool has_cached_samples = false;
    bool has_pending_batch = false;
};

inline bool AllAgentsAtFragmentBoundary(
    const std::vector<FragmentBoundaryState>& agents) {
    for (const auto& agent : agents) {
        if (agent.has_pending_action ||
            agent.has_cached_samples ||
            agent.has_pending_batch) {
            return false;
        }
    }
    return true;
}
