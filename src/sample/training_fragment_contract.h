#pragma once

#include "contracts/contract_namespaces.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

inline bool ValidateTrainingSampleAppend(
    const std::vector<training::Sample>& fragment,
    const training::Sample& sample,
    int expected_observation_dim,
    int expected_action_dim,
    std::string& error) {
    if (expected_observation_dim <= 0 || expected_action_dim <= 0 ||
        sample.observation_size() != expected_observation_dim ||
        sample.next_observation_size() != expected_observation_dim ||
        sample.action() < 0 || sample.action() >= expected_action_dim ||
        !std::isfinite(sample.reward()) ||
        !std::isfinite(sample.old_log_probability()) ||
        !std::isfinite(sample.old_value_prediction()) ||
        (sample.terminated() && sample.truncated())) {
        error = "sample shape, action, scalar, or end flags are invalid";
        return false;
    }
    for (int index = 0; index < expected_observation_dim; ++index) {
        if (!std::isfinite(sample.observation(index)) ||
            !std::isfinite(sample.next_observation(index))) {
            error = "sample observation contains a non-finite value";
            return false;
        }
    }

    const auto expected_end_kind =
        sample.terminated()
            ? training::TRANSITION_END_KIND_ENVIRONMENT_TERMINATED
            : sample.truncated()
                  ? training::TRANSITION_END_KIND_EXTERNAL_TRUNCATION
                  : training::TRANSITION_END_KIND_CONTINUING;
    if (sample.end_kind() != expected_end_kind) {
        error = "sample end kind conflicts with terminal flags";
        return false;
    }

    if (!fragment.empty()) {
        const auto& previous = fragment.back();
        if (previous.terminated() || previous.truncated()) {
            error = "fragment continues after an end transition";
            return false;
        }
        if (previous.action_step() == std::numeric_limits<std::uint64_t>::max() ||
            sample.action_step() != previous.action_step() + 1) {
            error = "sample action step is not contiguous";
            return false;
        }
        if (previous.next_observation_size() != sample.observation_size()) {
            error = "adjacent sample observation shapes differ";
            return false;
        }
        for (int index = 0; index < sample.observation_size(); ++index) {
            if (previous.next_observation(index) != sample.observation(index)) {
                error = "previous next observation differs from current observation";
                return false;
            }
        }
    }

    error.clear();
    return true;
}
