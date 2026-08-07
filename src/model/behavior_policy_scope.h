#pragma once

#include <vector>

enum class BehaviorPolicyScope {
    Unspecified,
    TrainingFragment,
    EvaluationEpisode,
};

struct ActiveBehaviorPolicyState {
    bool episode_active = false;
    BehaviorPolicyScope scope = BehaviorPolicyScope::Unspecified;
};

inline bool ActiveEpisodesAllowFragmentPolicySwitch(
    const std::vector<ActiveBehaviorPolicyState>& episodes) {
    for (const auto& episode : episodes) {
        if (episode.episode_active &&
            episode.scope != BehaviorPolicyScope::TrainingFragment) {
            return false;
        }
    }
    return true;
}
