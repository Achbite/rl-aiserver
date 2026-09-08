#pragma once

#include "policy/episode_mode.h"

#include <random>
#include <string>
#include <vector>

bool ValidateEpisodeModelOutput(const std::vector<float>& logits,
                                int expected_action_dim,
                                float value,
                                std::string& error);

bool SelectEpisodeAction(const std::vector<float>& logits,
                         const std::vector<bool>& action_mask,
                         PolicyMode mode,
                         double temperature,
                         std::mt19937& generator,
                         int& action,
                         float& log_probability,
                         std::string& error);
