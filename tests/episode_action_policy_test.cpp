#include "task/episode_action_policy.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

int main() {
    std::mt19937 generator(7);
    std::string error;
    int action = -1;
    float log_probability = 0.0f;
    const std::vector<float> logits{0.1f, 0.2f, 0.7f};
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
