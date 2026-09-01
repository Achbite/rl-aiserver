#include "task/episode_action_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

bool ValidateEpisodeModelOutput(const std::vector<float>& logits,
                                int expected_action_dim,
                                float value,
                                std::string& error) {
    if (expected_action_dim <= 0 ||
        logits.size() != static_cast<std::size_t>(expected_action_dim)) {
        error = "action logits do not match the expected action dimension";
        return false;
    }
    for (float logit : logits) {
        if (!std::isfinite(logit)) {
            error = "action logits contain a non-finite value";
            return false;
        }
    }
    if (!std::isfinite(value)) {
        error = "state value is non-finite";
        return false;
    }
    error.clear();
    return true;
}

bool SelectEpisodeAction(const std::vector<float>& logits,
                         const std::vector<bool>& action_mask,
                         maze::EpisodeMode mode,
                         double temperature,
                         std::mt19937& generator,
                         int& action,
                         float& log_probability,
                         std::string& error) {
    if (logits.empty() || !std::isfinite(temperature) ||
        temperature <= 0.0) {
        error = "action logits or temperature are invalid";
        return false;
    }
    for (float logit : logits) {
        if (!std::isfinite(logit)) {
            error = "action logits contain a non-finite value";
            return false;
        }
    }
    if (!action_mask.empty() &&
        (action_mask.size() != logits.size() ||
         std::none_of(action_mask.begin(), action_mask.end(),
                      [](bool available) { return available; }))) {
        error = "action mask does not define an available model action";
        return false;
    }
    const auto available = [&](std::size_t index) {
        return action_mask.empty() || action_mask[index];
    };

    if (mode == maze::EPISODE_MODE_EVALUATION) {
        std::size_t selected = logits.size();
        for (std::size_t index = 0; index < logits.size(); ++index) {
            if (available(index) &&
                (selected == logits.size() ||
                 logits[index] > logits[selected])) {
                selected = index;
            }
        }
        action = static_cast<int>(selected);
        log_probability = 0.0f;
        return true;
    }
    if (mode == maze::EPISODE_MODE_TRAINING) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < logits.size(); ++index) {
            if (available(index)) {
                maximum = std::max(
                    maximum, static_cast<double>(logits[index]));
            }
        }
        std::vector<double> probabilities;
        probabilities.reserve(logits.size());
        double sum = 0.0;
        for (std::size_t index = 0; index < logits.size(); ++index) {
            const double weight = available(index)
                ? std::exp(
                      (static_cast<double>(logits[index]) - maximum) /
                      temperature)
                : 0.0;
            probabilities.push_back(weight);
            sum += weight;
        }
        if (!std::isfinite(sum) || !(sum > 0.0)) {
            error = "softmax normalization is invalid";
            return false;
        }
        std::discrete_distribution<int> distribution(
            probabilities.begin(), probabilities.end());
        action = distribution(generator);
        log_probability = static_cast<float>(
            std::log(probabilities[static_cast<std::size_t>(action)] / sum));
        return true;
    }
    error = "Episode mode does not define a model action policy";
    return false;
}
