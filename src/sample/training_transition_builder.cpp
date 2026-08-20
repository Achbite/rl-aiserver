#include "sample/training_transition_builder.h"

#include <chrono>
#include <cmath>
#include <limits>

namespace {

bool FiniteVector(const std::vector<float>& values) {
    for (float value : values) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

}  // namespace

bool BuildRawRolloutTransition(
    const SessionManager::AgentRuntime& agent,
    const std::vector<float>& next_observation,
    const RewardDetail& reward,
    int expected_obs_dim,
    int expected_action_dim,
    SessionManager::RawRolloutTransition& transition,
    std::string& error) {
    transition = SessionManager::RawRolloutTransition{};
    if (!agent.has_pending_action || !agent.segment_open ||
        !agent.pinned_prepared_model.valid() || !reward.valid ||
        agent.pending_action_frame_id < 0 || agent.pending_action < 0 ||
        agent.pending_action >= expected_action_dim ||
        static_cast<int>(agent.pending_obs.size()) != expected_obs_dim ||
        static_cast<int>(next_observation.size()) != expected_obs_dim ||
        !FiniteVector(agent.pending_obs) ||
        !FiniteVector(next_observation) ||
        !std::isfinite(agent.pending_log_prob) ||
        !std::isfinite(agent.pending_value) ||
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
    transition.action_step =
        static_cast<uint64_t>(agent.pending_action_frame_id);
    transition.created_at_unix_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    error.clear();
    return true;
}

bool EstimateRolloutSegment(
    const std::vector<SessionManager::RawRolloutTransition>& segment,
    double gamma,
    double gae_lambda,
    double final_next_value,
    std::vector<float>& advantages,
    std::vector<float>& value_targets,
    std::string& error) {
    advantages.clear();
    value_targets.clear();
    if (segment.empty() || !std::isfinite(gamma) ||
        !std::isfinite(gae_lambda) || !std::isfinite(final_next_value)) {
        error = "rollout estimator inputs are empty or non-finite";
        return false;
    }

    advantages.resize(segment.size());
    value_targets.resize(segment.size());
    double next_advantage = 0.0;
    double next_value = final_next_value;
    for (std::size_t reverse = segment.size(); reverse > 0; --reverse) {
        const std::size_t index = reverse - 1;
        const auto& transition = segment[index];
        if (!std::isfinite(transition.reward) ||
            !std::isfinite(transition.behavior_value)) {
            error = "rollout segment contains a non-finite scalar";
            return false;
        }
        const double delta = static_cast<double>(transition.reward) +
                             gamma * next_value -
                             static_cast<double>(transition.behavior_value);
        const double advantage =
            delta + gamma * gae_lambda * next_advantage;
        const double value_target =
            advantage + static_cast<double>(transition.behavior_value);
        if (!std::isfinite(delta) || !std::isfinite(advantage) ||
            !std::isfinite(value_target) ||
            advantage < -std::numeric_limits<float>::max() ||
            advantage > std::numeric_limits<float>::max() ||
            value_target < -std::numeric_limits<float>::max() ||
            value_target > std::numeric_limits<float>::max()) {
            error = "rollout estimator produced a non-finite float result";
            return false;
        }
        advantages[index] = static_cast<float>(advantage);
        value_targets[index] = static_cast<float>(value_target);
        next_advantage = advantage;
        next_value = transition.behavior_value;
    }
    error.clear();
    return true;
}
