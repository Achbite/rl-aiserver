#pragma once

#include "maze/protocol/contract_types.h"

#include <string>
#include <vector>

struct ValidatedMazeMap {
    std::vector<bool> blocked;
    std::vector<int> geodesic_distance;
    int shortest_action_steps = -1;
    int max_finite_distance = -1;
};

bool ValidateMazeMapDescriptor(const maze::MapDescriptor& descriptor,
                               const std::string& expected_map_id,
                               ValidatedMazeMap& validated,
                               std::string& error);
