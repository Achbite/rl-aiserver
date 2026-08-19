#include "task/maze_map_contract.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

maze::MapDescriptor MakeDescriptor(int cols, int rows,
                                   int start_x, int start_y,
                                   int goal_x, int goal_y,
                                   std::string bitmap,
                                   int shortest_action_steps) {
    maze::MapDescriptor descriptor;
    descriptor.set_map_id("test-map");
    descriptor.set_format_version(4);
    descriptor.set_grid_columns(cols);
    descriptor.set_grid_rows(rows);
    descriptor.set_grid_size_microunits(1000000);
    descriptor.set_start_grid_x(start_x);
    descriptor.set_start_grid_y(start_y);
    descriptor.set_goal_grid_x(goal_x);
    descriptor.set_goal_grid_y(goal_y);
    descriptor.set_action_rule_id("maze.action.9-way.no-corner-cut.v1");
    descriptor.set_blocked_bitmap(std::move(bitmap));
    descriptor.set_shortest_action_steps(shortest_action_steps);
    std::string error;
    descriptor.mutable_canonical_digest()->set_algorithm(
        common::DIGEST_ALGORITHM_SHA256);
    descriptor.mutable_canonical_digest()->set_hex(
        CanonicalMazeMapChecksum(descriptor, error));
    Require(error.empty() &&
                descriptor.canonical_digest().hex().size() == 64,
            "test descriptor checksum must be generated");
    return descriptor;
}

void TestOpenGridContract() {
    auto descriptor = MakeDescriptor(
        3, 3, 0, 0, 2, 2, std::string(9, '\0'), 2);
    ValidatedMazeMap validated;
    std::string error;
    Require(ValidateMazeMapDescriptor(
                descriptor, "test-map", descriptor.canonical_digest().hex(),
                validated, error),
            "valid open grid must be accepted: " + error);
    Require(validated.checksum_sha256 ==
                descriptor.canonical_digest().hex(),
            "validated map preserves its assigned digest");
}

void TestIdentityFailures() {
    auto descriptor = MakeDescriptor(
        3, 3, 0, 0, 2, 2, std::string(9, '\0'), 2);
    ValidatedMazeMap validated;
    std::string error;
    Require(!ValidateMazeMapDescriptor(
                descriptor, "wrong-map", descriptor.canonical_digest().hex(),
                validated, error),
            "wrong map id must be rejected");

    error.clear();
    Require(!ValidateMazeMapDescriptor(
                descriptor, "test-map", std::string(64, '0'),
                validated, error),
            "wrong assigned checksum must be rejected");

}

void TestBitmapValuesAreBoolean() {
    std::string bitmap(9, '\0');
    bitmap[4] = '\2';
    maze::MapDescriptor descriptor = MakeDescriptor(
        3, 3, 0, 0, 2, 2, std::string(9, '\0'), 2);
    descriptor.set_blocked_bitmap(std::move(bitmap));
    std::string error;
    Require(CanonicalMazeMapChecksum(descriptor, error).empty(),
            "non-boolean bitmap values must not be checksummed");
}

}  // namespace

int main() {
    TestOpenGridContract();
    TestIdentityFailures();
    TestBitmapValuesAreBoolean();
    std::cout << "maze_map_contract: PASS" << std::endl;
    return 0;
}
