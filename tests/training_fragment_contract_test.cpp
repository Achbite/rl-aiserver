#include "sample/training_fragment_contract.h"
#include "sample/training_transition_builder.h"

#include <cassert>
#include <string>
#include <vector>

namespace {

training::Sample Sample(std::uint64_t step, float observation) {
    training::Sample sample;
    sample.set_action(3);
    sample.set_reward(0.0f);
    sample.set_old_log_probability(-0.5f);
    sample.set_old_value_prediction(0.75f);
    sample.set_end_kind(training::TRANSITION_END_KIND_CONTINUING);
    sample.set_action_step(step);
    for (int index = 0; index < 17; ++index) {
        sample.add_observation(observation + static_cast<float>(index));
        sample.add_next_observation(
            observation + 1.0f + static_cast<float>(index));
    }
    return sample;
}

}  // namespace

int main() {
    SessionManager::AgentRuntime agent;
    agent.has_pending_action = true;
    agent.pending_action = 4;
    agent.pending_action_frame_id = 12;
    agent.pending_log_prob = -1.25f;
    agent.pending_value = 0.5f;
    for (int index = 0; index < 17; ++index) {
        agent.pending_obs.push_back(static_cast<float>(index));
    }
    std::vector<float> next_observation(17, 1.0f);
    RewardDetail reward;
    reward.total = 0.0f;
    reward.task_total = 0.0f;
    reward.shaping_total = 0.0f;
    training::Sample built;
    std::string error;
    const std::vector<training::Sample> empty_fragment;
    assert(BuildTrainingSample(
        agent, next_observation, reward, false,
        maze::MAZE_TERMINATION_REASON_ACTIVE, 17, 9,
        empty_fragment, built, error));
    assert(built.observation_size() == 17);
    assert(built.next_observation_size() == 17);
    assert(built.action() == 4);
    assert(built.reward() == 0.0f);
    assert(built.action_step() == 12);
    assert(built.end_kind() ==
           training::TRANSITION_END_KIND_CONTINUING);

    std::vector<training::Sample> fragment;
    auto first = Sample(40, 2.0f);
    assert(ValidateTrainingSampleAppend(fragment, first, 17, 9, error));
    fragment.push_back(first);
    auto second = Sample(41, 3.0f);
    assert(ValidateTrainingSampleAppend(fragment, second, 17, 9, error));

    auto discontinuous_step = second;
    discontinuous_step.set_action_step(42);
    assert(!ValidateTrainingSampleAppend(
        fragment, discontinuous_step, 17, 9, error));

    auto wrong_shape = second;
    wrong_shape.mutable_next_observation()->RemoveLast();
    assert(!ValidateTrainingSampleAppend(
        fragment, wrong_shape, 17, 9, error));

    auto terminal = second;
    terminal.set_terminated(true);
    terminal.set_end_kind(
        training::TRANSITION_END_KIND_ENVIRONMENT_TERMINATED);
    assert(ValidateTrainingSampleAppend(fragment, terminal, 17, 9, error));
    return 0;
}
