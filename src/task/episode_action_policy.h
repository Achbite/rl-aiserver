#pragma once

#include "contracts/contract_namespaces.h"

#include <random>
#include <string>
#include <vector>

bool SelectEpisodeAction(const std::vector<float>& logits,
                         maze::EpisodeMode mode,
                         double temperature,
                         std::mt19937& generator,
                         int& action,
                         float& log_probability,
                         std::string& error);
