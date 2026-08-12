#include "sample/training_transition_builder.h"

#include "sample/training_fragment_contract.h"

#include <cmath>

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
    std::string& error) {
    sample.Clear();
    if (!agent.has_pending_action || !reward.valid ||
        agent.pending_action_frame_id < 0 ||
        static_cast<int>(agent.pending_obs.size()) != expected_obs_dim ||
        static_cast<int>(next_observation.size()) != expected_obs_dim) {
        error = "pending transition inputs are incomplete";
        return false;
    }
    if (is_done !=
        (reason == maze::MAZE_TERMINATION_REASON_GOAL_REACHED ||
         reason == maze::MAZE_TERMINATION_REASON_TIME_LIMIT)) {
        error = "pending transition termination is inconsistent";
        return false;
    }

    for (float value : agent.pending_obs) sample.add_observation(value);
    for (float value : next_observation) {
        sample.add_next_observation(value);
    }
    sample.set_action(agent.pending_action);
    sample.set_reward(reward.total);
    sample.set_old_log_probability(agent.pending_log_prob);
    sample.set_old_value_prediction(agent.pending_value);
    sample.set_terminated(is_done);
    sample.set_truncated(false);
    sample.set_end_kind(
        is_done
            ? training::TRANSITION_END_KIND_ENVIRONMENT_TERMINATED
            : training::TRANSITION_END_KIND_CONTINUING);
    sample.set_action_step(
        static_cast<uint64_t>(agent.pending_action_frame_id));

    if (!ValidateTrainingSampleAppend(
            fragment, sample, expected_obs_dim, expected_action_dim, error)) {
        return false;
    }
    error.clear();
    return true;
}
