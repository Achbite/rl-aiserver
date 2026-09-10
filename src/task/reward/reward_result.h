#pragma once

#include <string>
#include <utility>
#include <vector>

struct RewardResult {
    bool valid = true;
    std::string error;
    float total = 0.0f;
    std::vector<std::pair<std::string, float>> items;
};
