#pragma once

#include "contracts/contract_namespaces.h"

#include <string>
#include <vector>

struct ValidatedMazeMap {
    std::vector<bool> blocked;
    std::vector<int> geodesic_distance;
    int shortest_action_steps = -1;
    int max_finite_distance = -1;
    std::string checksum_sha256;
};

std::string CanonicalMazeMapChecksum(const maze::MapDescriptor& descriptor,
                                     std::string& error);

bool ValidateMazeMapDescriptor(const maze::MapDescriptor& descriptor,
                               const std::string& expected_map_id,
                               const std::string& expected_checksum,
                               ValidatedMazeMap& validated,
                               std::string& error);
