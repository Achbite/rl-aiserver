#include "model/fragment_boundary.h"

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
    return 0;
}
