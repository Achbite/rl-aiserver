#include "sample/training_transition_builder.h"

#include <algorithm>
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

bool IsSha256Digest(const common::ContentDigest& digest) {
    if (digest.algorithm() != common::DIGEST_ALGORITHM_SHA256 ||
        digest.hex().size() != 64) {
        return false;
    }
    return std::all_of(digest.hex().begin(), digest.hex().end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
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

bool ProjectProcessedSegment(
    const std::vector<SessionManager::RawRolloutTransition>& raw_segment,
    const std::vector<float>& advantages,
    const std::vector<float>& value_targets,
    const std::string& segment_id,
    const training::ModelIdentity& behavior_model,
    int observation_dimension,
    int action_count,
    std::vector<training::ProcessedTransition>& processed,
    std::string& error) {
    processed.clear();
    if (raw_segment.empty() || advantages.size() != raw_segment.size() ||
        value_targets.size() != raw_segment.size() || segment_id.empty() ||
        observation_dimension <= 0 || action_count <= 0 ||
        behavior_model.model_lineage_id().empty() ||
        !behavior_model.has_model_step() ||
        !IsSha256Digest(behavior_model.artifact_digest()) ||
        !IsSha256Digest(behavior_model.manifest_digest())) {
        error = "processed segment identity or estimator output is incomplete";
        return false;
    }

    processed.reserve(raw_segment.size());
    uint64_t expected_action_step = raw_segment.front().action_step;
    for (std::size_t index = 0; index < raw_segment.size(); ++index) {
        const auto& raw = raw_segment[index];
        if (raw.observation.size() !=
                static_cast<std::size_t>(observation_dimension) ||
            raw.next_observation.size() !=
                static_cast<std::size_t>(observation_dimension) ||
            raw.action < 0 || raw.action >= action_count ||
            raw.action_step != expected_action_step ||
            raw.created_at_unix_ms <= 0 || !FiniteVector(raw.observation) ||
            !FiniteVector(raw.next_observation) ||
            !std::isfinite(raw.reward) ||
            !std::isfinite(raw.behavior_log_probability) ||
            !std::isfinite(raw.behavior_value) ||
            !std::isfinite(advantages[index]) ||
            !std::isfinite(value_targets[index]) ||
            (index + 1 < raw_segment.size() &&
             raw.next_observation != raw_segment[index + 1].observation)) {
            error = "processed segment contains an invalid transition";
            processed.clear();
            return false;
        }

        training::ProcessedTransition item;
        item.set_item_id(segment_id + "/transition-" +
                         std::to_string(index));
        for (float value : raw.observation) item.add_observation(value);
        item.set_action(raw.action);
        item.set_behavior_log_probability(raw.behavior_log_probability);
        item.set_behavior_value(raw.behavior_value);
        item.set_advantage(advantages[index]);
        item.set_value_target(value_targets[index]);
        item.set_behavior_model_step(behavior_model.model_step());
        item.set_created_at_unix_ms(raw.created_at_unix_ms);
        processed.push_back(std::move(item));
        ++expected_action_step;
    }
    error.clear();
    return true;
}
