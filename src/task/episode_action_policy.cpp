#include "task/episode_action_policy.h"

#include <algorithm>
#include <cmath>
#include <numeric>

bool SelectEpisodeAction(const std::vector<float>& logits,
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

    if (mode == maze::EPISODE_MODE_EVALUATION_ARGMAX) {
        action = static_cast<int>(std::distance(
            logits.begin(),
            std::max_element(logits.begin(), logits.end())));
        log_probability = 0.0f;
        return true;
    }
    if (mode == maze::EPISODE_MODE_TRAINING ||
        mode == maze::EPISODE_MODE_EVALUATION_STOCHASTIC) {
        const double maximum = *std::max_element(logits.begin(), logits.end());
        std::vector<double> probabilities;
        probabilities.reserve(logits.size());
        double sum = 0.0;
        for (float logit : logits) {
            const double weight = std::exp(
                (static_cast<double>(logit) - maximum) / temperature);
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
