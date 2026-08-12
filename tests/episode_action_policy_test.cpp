#include "task/episode_action_policy.h"

#include <cassert>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string Hex(const std::string& value) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (unsigned char byte : value) {
        stream << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return stream.str();
}

}  // namespace

int main() {
    std::mt19937 generator(7);
    std::string error;
    int action = -1;
    float log_probability = 0.0f;
    const std::vector<float> logits{0.1f, 0.2f, 0.7f};
    assert(ValidateEpisodeModelOutput(logits, 3, 0.25f, error));
    assert(error.empty());
    assert(!ValidateEpisodeModelOutput(logits, 2, 0.25f, error));
    assert(!ValidateEpisodeModelOutput(
        logits, 3, std::numeric_limits<float>::quiet_NaN(), error));
    assert(!ValidateEpisodeModelOutput(
        logits, 3, std::numeric_limits<float>::infinity(), error));
    assert(!ValidateEpisodeModelOutput(
        logits, 3, -std::numeric_limits<float>::infinity(), error));

    // Shared actor-wire fixture a3-arch-policy-wire-v1. The Learner consumes
    // the same float32 logits, value, actions, and old log probabilities.
    const std::vector<float> golden_logits{
        0.0f, std::log(2.0f), std::log(4.0f),
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const std::array<float, 9> golden_log_probabilities{
        -2.5649492740631104f,
        -1.8718022108078003f,
        -1.1786550283432007f,
        -2.5649492740631104f,
        -2.5649492740631104f,
        -2.5649492740631104f,
        -2.5649492740631104f,
        -2.5649492740631104f,
        -2.5649492740631104f,
    };
    assert(ValidateEpisodeModelOutput(
        golden_logits, 9, 0.25f, error));
    std::array<bool, 9> golden_actions_seen{};
    std::mt19937 golden_generator(17);
    for (int i = 0; i < 4096; ++i) {
        assert(SelectEpisodeAction(
            golden_logits, maze::EPISODE_MODE_TRAINING, 1.0,
            golden_generator, action, log_probability, error));
        assert(action >= 0 && action < 9);
        assert(std::fabs(
                   log_probability -
                   golden_log_probabilities[static_cast<std::size_t>(action)]) <
               1e-6f);
        golden_actions_seen[static_cast<std::size_t>(action)] = true;
    }
    for (bool seen : golden_actions_seen) assert(seen);

    const std::vector<float> golden_observation{
        0.0f, std::log(2.0f), std::log(4.0f), 0.25f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    training::Sample wire_sample;
    for (float item : golden_observation) {
        wire_sample.add_observation(item);
        wire_sample.add_next_observation(item);
    }
    wire_sample.set_action(2);
    wire_sample.set_old_log_probability(golden_log_probabilities[2]);
    wire_sample.set_old_value_prediction(0.25f);
    wire_sample.set_end_kind(
        training::TRANSITION_END_KIND_CONTINUING);
    const std::string expected_wire_hex =
        "0a44000000001872313f1872b13f0000803e0000000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000000000000"
        "0000000000001244000000001872313f1872b13f0000803e0000000000000000"
        "0000000000000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000018022d2bde96bf350000803e4801";
    const std::string wire_bytes = wire_sample.SerializeAsString();
    assert(Hex(wire_bytes) == expected_wire_hex);
    training::Sample parsed_wire_sample;
    assert(parsed_wire_sample.ParseFromString(wire_bytes));
    assert(parsed_wire_sample.action() == 2);
    assert(parsed_wire_sample.old_log_probability() ==
           golden_log_probabilities[2]);
    assert(parsed_wire_sample.old_value_prediction() == 0.25f);

    for (int i = 0; i < 100; ++i) {
        assert(SelectEpisodeAction(
            logits, maze::EPISODE_MODE_EVALUATION_ARGMAX, 1.0,
            generator, action, log_probability, error));
        assert(action == 2);
        assert(log_probability == 0.0f);
    }

    bool observed_non_argmax = false;
    for (int i = 0; i < 200; ++i) {
        assert(SelectEpisodeAction(
            logits, maze::EPISODE_MODE_EVALUATION_STOCHASTIC, 1.0,
            generator, action, log_probability, error));
        observed_non_argmax = observed_non_argmax || action != 2;
        assert(std::isfinite(log_probability));
    }
    assert(observed_non_argmax);

    assert(!SelectEpisodeAction(
        logits, maze::EPISODE_MODE_UNSPECIFIED, 1.0,
        generator, action, log_probability, error));
    assert(!SelectEpisodeAction(
        logits, maze::EPISODE_MODE_TRAINING, 0.0,
        generator, action, log_probability, error));
    assert(!SelectEpisodeAction(
        {0.0f, std::numeric_limits<float>::quiet_NaN()},
        maze::EPISODE_MODE_TRAINING, 1.0,
        generator, action, log_probability, error));

    std::mt19937 cold_generator(11);
    std::mt19937 hot_generator(11);
    int cold_argmax_count = 0;
    int hot_argmax_count = 0;
    for (int i = 0; i < 5000; ++i) {
        assert(SelectEpisodeAction(
            logits, maze::EPISODE_MODE_TRAINING, 0.25,
            cold_generator, action, log_probability, error));
        cold_argmax_count += action == 2 ? 1 : 0;
        assert(SelectEpisodeAction(
            logits, maze::EPISODE_MODE_TRAINING, 2.0,
            hot_generator, action, log_probability, error));
        hot_argmax_count += action == 2 ? 1 : 0;
    }
    assert(cold_argmax_count > hot_argmax_count);

    std::cout << "episode action policy contract passed\n";
    return 0;
}
