#include "task/episode_action_policy.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

int main() {
    const std::vector<float> logits{0.1f, 0.2f, 0.7f};
    std::string error;
    assert(ValidateEpisodeModelOutput(logits, 3, 0.25f, error));

    std::mt19937 generator(7);
    int action = -1;
    float log_probability = 0.0f;
    assert(SelectEpisodeAction(
        logits, maze::EPISODE_MODE_TRAINING, 1.0,
        generator, action, log_probability, error));
    assert(action >= 0 && action < 3);
    assert(std::isfinite(log_probability));

    training::Sample sample;
    sample.set_action(action);
    sample.set_reward(0.0f);
    sample.set_old_log_probability(log_probability);
    sample.set_old_value_prediction(0.25f);
    sample.set_end_kind(training::TRANSITION_END_KIND_CONTINUING);
    const std::string bytes = sample.SerializeAsString();
    training::Sample parsed;
    assert(parsed.ParseFromString(bytes));
    assert(parsed.action() == action);
    assert(parsed.old_log_probability() == log_probability);

    assert(SelectEpisodeAction(
        logits, maze::EPISODE_MODE_EVALUATION, 1.0,
        generator, action, log_probability, error));
    assert(action == 2 && log_probability == 0.0f);

    assert(!ValidateEpisodeModelOutput(
        logits, 2, 0.25f, error));
    assert(!ValidateEpisodeModelOutput(
        logits, 3, std::numeric_limits<float>::quiet_NaN(), error));
    assert(!SelectEpisodeAction(
        logits, maze::EPISODE_MODE_UNSPECIFIED, 1.0,
        generator, action, log_probability, error));

    std::cout << "episode_action_policy_contract: PASS\n";
    return 0;
}
