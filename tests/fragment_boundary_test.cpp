#include "model/fragment_boundary.h"
#include "model/model_boundary.h"

#include <iostream>
#include <vector>

int main() {
    std::vector<FragmentBoundaryState> agents(4);
    if (!AllAgentsAtFragmentBoundary(agents)) {
        std::cerr << "four idle agents must form a global boundary\n";
        return 1;
    }

    agents[2].has_pending_action = true;
    if (AllAgentsAtFragmentBoundary(agents)) {
        std::cerr << "a pending action must block a model switch\n";
        return 1;
    }
    agents[2].has_pending_action = false;
    agents[1].has_cached_samples = true;
    if (AllAgentsAtFragmentBoundary(agents)) {
        std::cerr << "cached samples must block a model switch\n";
        return 1;
    }
    agents[1].has_cached_samples = false;
    agents[3].has_pending_batch = true;
    if (AllAgentsAtFragmentBoundary(agents)) {
        std::cerr << "an unsent fragment must block a model switch\n";
        return 1;
    }

    std::unordered_map<int, int64_t> closure_counts;
    for (int version = 0; version < 6; ++version) {
        closure_counts[version] = 512;
    }
    if (SelectModelBoundaryTarget(0, 512, closure_counts) != 6) {
        std::cerr << "exact closure must wait for every owned update\n";
        return 1;
    }
    std::unordered_map<int, int64_t> curriculum_counts;
    for (int version = 0; version < 22; ++version) {
        curriculum_counts[version] = 512;
    }
    curriculum_counts[22] = 384;
    curriculum_counts[23] = 384;
    if (SelectModelBoundaryTarget(0, 512, curriculum_counts) != 22) {
        std::cerr <<
            "curriculum must not infer an unavailable model from produced samples\n";
        return 1;
    }
    if (SelectPerAgentFragmentSamples(384, 512, 4, 128, true) != 32) {
        std::cerr << "four active agents must fill a 128-sample remainder\n";
        return 1;
    }
    if (SelectPerAgentFragmentSamples(0, 512, 4, 128, true) != 128 ||
        SelectPerAgentFragmentSamples(385, 512, 4, 128, true) != 128 ||
        SelectPerAgentFragmentSamples(384, 512, 4, 128, false) != 128) {
        std::cerr << "unsafe fragment adjustments must keep the configured size\n";
        return 1;
    }
    if (!IsModelSampleBoundary(6144, 512) ||
        IsModelSampleBoundary(6112, 512) ||
        IsModelSampleBoundary(0, 512) ||
        IsModelSampleBoundary(512, 0)) {
        std::cerr << "model sample boundary detection is inconsistent\n";
        return 1;
    }
    if (!ShouldFlushAgentFragment(32, 128, 6144, 512)) {
        std::cerr <<
            "partial fragments must flush before waiting for the next model\n";
        return 1;
    }
    if (ShouldFlushAgentFragment(32, 128, 6112, 512) ||
        ShouldFlushAgentFragment(0, 128, 6144, 512) ||
        ShouldFlushAgentFragment(32, 0, 6144, 512)) {
        std::cerr << "non-boundary or invalid fragments must not flush early\n";
        return 1;
    }
    return 0;
}
