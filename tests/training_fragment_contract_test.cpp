#include "sample/training_fragment_contract.h"
#include "sample/training_transition_builder.h"

#include <openssl/evp.h>

#include <cassert>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

training::Sample Sample(std::uint64_t step, float observation) {
    training::Sample sample;
    sample.set_action(3);
    sample.set_reward(0.25f);
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

std::string Sha256(const std::string& bytes) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_size = 0;
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    assert(context != nullptr);
    assert(EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1);
    assert(EVP_DigestUpdate(context, bytes.data(), bytes.size()) == 1);
    assert(EVP_DigestFinal_ex(context, digest, &digest_size) == 1);
    EVP_MD_CTX_free(context);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index) {
        output << std::setw(2) << static_cast<int>(digest[index]);
    }
    return output.str();
}

void TestProductionTrainingSampleBuilder() {
    SessionManager::AgentRuntime agent;
    agent.has_pending_action = true;
    agent.pending_action = 4;
    agent.pending_action_frame_id = 12;
    agent.pending_log_prob = -1.25f;
    agent.pending_value = 0.5f;
    for (int index = 0; index < 17; ++index) {
        agent.pending_obs.push_back(static_cast<float>(index) * 0.25f);
    }
    std::vector<float> next_observation;
    for (int index = 0; index < 17; ++index) {
        next_observation.push_back(
            100.0f + static_cast<float>(index) * 0.5f);
    }
    RewardDetail reward;
    reward.total = 1.75f;
    reward.task_total = 0.0f;
    reward.shaping_total = 1.75f;

    std::string error;
    training::Sample built;
    const std::vector<training::Sample> empty_fragment;
    assert(BuildTrainingSample(
        agent, next_observation, reward, false,
        maze::MAZE_TERMINATION_REASON_ACTIVE, 17, 9,
        empty_fragment, built, error));
    assert(built.observation_size() == 17);
    assert(built.next_observation_size() == 17);
    assert(built.action() == 4);
    assert(built.reward() == 1.75f);
    assert(built.old_log_probability() == -1.25f);
    assert(built.old_value_prediction() == 0.5f);
    assert(!built.terminated() && !built.truncated());
    assert(built.end_kind() ==
           training::TRANSITION_END_KIND_CONTINUING);
    assert(built.action_step() == 12);
    assert(Sha256(built.SerializeAsString()) ==
           "41a98a43cef1970187f41a6b407a4e976798546cc487ad0a5d96ab1171d7fbd6");

    agent.pending_obs = next_observation;
    agent.pending_action_frame_id = 13;
    std::vector<float> terminal_observation = next_observation;
    terminal_observation[0] += 1.0f;
    reward.total = -2.0f;
    std::vector<training::Sample> fragment{built};
    training::Sample terminal;
    assert(BuildTrainingSample(
        agent, terminal_observation, reward, true,
        maze::MAZE_TERMINATION_REASON_TIME_LIMIT, 17, 9,
        fragment, terminal, error));
    assert(terminal.terminated() && !terminal.truncated());
    assert(terminal.end_kind() ==
           training::TRANSITION_END_KIND_ENVIRONMENT_TERMINATED);
    assert(terminal.action_step() == 13);
}

}  // namespace

int main() {
    TestProductionTrainingSampleBuilder();
    std::string error;
    std::vector<training::Sample> fragment;
    auto first = Sample(40, 2.0f);
    assert(ValidateTrainingSampleAppend(fragment, first, 17, 9, error));
    fragment.push_back(first);

    auto second = Sample(41, 3.0f);
    assert(ValidateTrainingSampleAppend(fragment, second, 17, 9, error));

    const auto rejected = [&](const training::Sample& sample) {
        assert(!ValidateTrainingSampleAppend(fragment, sample, 17, 9, error));
    };

    auto discontinuous_observation = second;
    discontinuous_observation.set_observation(6, -10.0f);
    rejected(discontinuous_observation);

    auto discontinuous_step = second;
    discontinuous_step.set_action_step(42);
    rejected(discontinuous_step);

    auto negative_action = second;
    negative_action.set_action(-1);
    rejected(negative_action);

    auto action_past_dimension = second;
    action_past_dimension.set_action(9);
    rejected(action_past_dimension);

    auto non_finite_reward = second;
    non_finite_reward.set_reward(std::numeric_limits<float>::quiet_NaN());
    rejected(non_finite_reward);
    non_finite_reward.set_reward(std::numeric_limits<float>::infinity());
    rejected(non_finite_reward);

    auto non_finite_log_probability = second;
    non_finite_log_probability.set_old_log_probability(
        std::numeric_limits<float>::quiet_NaN());
    rejected(non_finite_log_probability);
    non_finite_log_probability.set_old_log_probability(
        -std::numeric_limits<float>::infinity());
    rejected(non_finite_log_probability);

    auto non_finite_value = second;
    non_finite_value.set_old_value_prediction(
        std::numeric_limits<float>::quiet_NaN());
    rejected(non_finite_value);
    non_finite_value.set_old_value_prediction(
        std::numeric_limits<float>::infinity());
    rejected(non_finite_value);

    auto non_finite_observation = second;
    non_finite_observation.set_observation(
        5, std::numeric_limits<float>::quiet_NaN());
    rejected(non_finite_observation);
    non_finite_observation.set_observation(
        5, -std::numeric_limits<float>::infinity());
    rejected(non_finite_observation);

    auto non_finite_next_observation = second;
    non_finite_next_observation.set_next_observation(
        5, std::numeric_limits<float>::quiet_NaN());
    rejected(non_finite_next_observation);
    non_finite_next_observation.set_next_observation(
        5, std::numeric_limits<float>::infinity());
    rejected(non_finite_next_observation);

    auto wrong_shape = second;
    wrong_shape.mutable_next_observation()->RemoveLast();
    rejected(wrong_shape);

    auto conflicting_end_flags = second;
    conflicting_end_flags.set_terminated(true);
    conflicting_end_flags.set_truncated(true);
    conflicting_end_flags.set_end_kind(
        training::TRANSITION_END_KIND_ENVIRONMENT_TERMINATED);
    rejected(conflicting_end_flags);

    auto conflicting_end_kind = second;
    conflicting_end_kind.set_end_kind(
        training::TRANSITION_END_KIND_ENVIRONMENT_TERMINATED);
    rejected(conflicting_end_kind);

    conflicting_end_kind = second;
    conflicting_end_kind.set_terminated(true);
    rejected(conflicting_end_kind);

    conflicting_end_kind = second;
    conflicting_end_kind.set_truncated(true);
    conflicting_end_kind.set_end_kind(
        training::TRANSITION_END_KIND_ENVIRONMENT_TERMINATED);
    rejected(conflicting_end_kind);

    auto terminal = second;
    terminal.set_terminated(true);
    terminal.set_end_kind(
        training::TRANSITION_END_KIND_ENVIRONMENT_TERMINATED);
    assert(ValidateTrainingSampleAppend(fragment, terminal, 17, 9, error));

    auto truncated = second;
    truncated.set_truncated(true);
    truncated.set_end_kind(
        training::TRANSITION_END_KIND_EXTERNAL_TRUNCATION);
    assert(ValidateTrainingSampleAppend(fragment, truncated, 17, 9, error));

    std::vector<training::Sample> ended_fragment{terminal};
    assert(!ValidateTrainingSampleAppend(
        ended_fragment, second, 17, 9, error));

    ended_fragment = {truncated};
    assert(!ValidateTrainingSampleAppend(
        ended_fragment, second, 17, 9, error));

    return 0;
}
