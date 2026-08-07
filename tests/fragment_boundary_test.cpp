#include "model/behavior_policy_scope.h"
#include "model/fragment_boundary.h"
#include "model/model_boundary.h"

#include <iostream>
#include <vector>

int main() {
    std::vector<ActiveBehaviorPolicyState> episodes;
    if (!ActiveEpisodesAllowFragmentPolicySwitch(episodes)) {
        std::cerr << "no active episode must allow a staged model switch\n";
        return 1;
    }
    episodes.push_back(ActiveBehaviorPolicyState{
        true, BehaviorPolicyScope::TrainingFragment});
    if (!ActiveEpisodesAllowFragmentPolicySwitch(episodes)) {
        std::cerr << "a training fragment-scoped episode must allow switching\n";
        return 1;
    }
    episodes.push_back(ActiveBehaviorPolicyState{
        true, BehaviorPolicyScope::EvaluationEpisode});
    if (ActiveEpisodesAllowFragmentPolicySwitch(episodes)) {
        std::cerr << "an evaluation episode must block a staged model switch\n";
        return 1;
    }
    episodes.back().episode_active = false;
    if (!ActiveEpisodesAllowFragmentPolicySwitch(episodes)) {
        std::cerr << "an inactive evaluation episode must not block switching\n";
        return 1;
    }
    episodes.push_back(ActiveBehaviorPolicyState{
        true, BehaviorPolicyScope::Unspecified});
    if (ActiveEpisodesAllowFragmentPolicySwitch(episodes)) {
        std::cerr << "an unspecified active scope must fail closed\n";
        return 1;
    }

    std::vector<FragmentBoundaryState> agents(4);
    if (!AllLocalAgentsAtFragmentBoundary(agents)) {
        std::cerr << "four idle agents must form a pod-local boundary\n";
        return 1;
    }

    agents[2].has_pending_action = true;
    if (AllLocalAgentsAtFragmentBoundary(agents)) {
        std::cerr << "a pending action must block a model switch\n";
        return 1;
    }
    agents[2].has_pending_action = false;
    agents[1].has_cached_samples = true;
    if (AllLocalAgentsAtFragmentBoundary(agents)) {
        std::cerr << "cached samples must block a model switch\n";
        return 1;
    }
    agents[1].has_cached_samples = false;
    agents[3].has_pending_batch = true;
    if (AllLocalAgentsAtFragmentBoundary(agents)) {
        std::cerr << "an unsent fragment must block a model switch\n";
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
    if (!ShouldFlushAgentFragment(32, 128, true)) {
        std::cerr <<
            "a staged model must close every non-empty Agent fragment\n";
        return 1;
    }
    if (!ShouldFlushAgentFragment(128, 128, false) ||
        ShouldFlushAgentFragment(32, 128, false) ||
        ShouldFlushAgentFragment(0, 128, true) ||
        ShouldFlushAgentFragment(32, 0, true)) {
        std::cerr << "fragment flush conditions are inconsistent\n";
        return 1;
    }
    return 0;
}
