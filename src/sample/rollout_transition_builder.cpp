#include "sample/rollout_transition_builder.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace {

bool FiniteVector(const std::vector<float>& values) {
    for (float value : values) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

}  // namespace

bool BuildRawRolloutTransition(
    const AgentTrainingState& agent,
    const std::vector<float>& next_observation,
    const RewardResult& reward,
    int expected_obs_dim,
    int expected_action_dim,
    const std::string& action_mask_mode,
    RawRolloutTransition& transition,
    std::string& error) {
    transition = RawRolloutTransition{};
    const bool mask_valid =
        (action_mask_mode == "disabled" &&
         agent.pending_action_mask.empty()) ||
        (action_mask_mode == "required" &&
         agent.pending_action_mask.size() ==
             static_cast<std::size_t>(expected_action_dim) &&
         std::any_of(agent.pending_action_mask.begin(),
                     agent.pending_action_mask.end(),
                     [](bool available) { return available; }) &&
         agent.pending_action >= 0 &&
         agent.pending_action < expected_action_dim &&
         agent.pending_action_mask[
             static_cast<std::size_t>(agent.pending_action)]);
    if (!agent.has_pending_action || !agent.segment_open ||
        !agent.pinned_prepared_model.valid() || !reward.valid ||
        agent.pending_action_frame_id < 0 || agent.pending_action < 0 ||
        agent.pending_action >= expected_action_dim ||
        static_cast<int>(agent.pending_obs.size()) != expected_obs_dim ||
        static_cast<int>(next_observation.size()) != expected_obs_dim ||
        !FiniteVector(agent.pending_obs) ||
        !FiniteVector(next_observation) ||
        !std::isfinite(agent.pending_log_prob) ||
        !std::isfinite(agent.pending_value) || !mask_valid ||
        !std::isfinite(reward.total)) {
        error = "trusted rollout transition inputs are incomplete or non-finite";
        return false;
    }
    if (!agent.segment_transitions.empty() &&
        static_cast<uint64_t>(agent.pending_action_frame_id) !=
            agent.segment_transitions.back().action_step + 1) {
        error = "single-Agent segment action steps are not contiguous";
        return false;
    }

    transition.observation = agent.pending_obs;
    transition.next_observation = next_observation;
    transition.action = agent.pending_action;
    transition.reward = reward.total;
    transition.behavior_log_probability = agent.pending_log_prob;
    transition.behavior_value = agent.pending_value;
    transition.action_mask = agent.pending_action_mask;
    transition.action_step =
        static_cast<uint64_t>(agent.pending_action_frame_id);
    transition.created_at_unix_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    error.clear();
    return true;
}
