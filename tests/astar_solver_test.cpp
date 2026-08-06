#include "ai/astar_solver.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void TestUnitCostShortestPath() {
    AStarSolver solver;
    Require(solver.InitGrid(3, 3, std::vector<bool>(9, false)),
            "valid authoritative bitmap must initialize the solver");
    Require(solver.PlanPath(0, 0, 2, 2),
            "open grid path must be found");
    Require(solver.GetPathLength() == 3,
            "two unit-cost diagonal actions must produce three path nodes");
    Require(solver.GetAction(0, 0) == 2,
            "first action on open diagonal must be up-right");
}

void TestCornerCutRejected() {
    std::vector<bool> blocked(4, false);
    blocked[1] = true;
    blocked[2] = true;
    AStarSolver solver;
    Require(solver.InitGrid(2, 2, blocked),
            "corner fixture bitmap must initialize the solver");
    Require(!solver.PlanPath(0, 0, 1, 1),
            "solver must not cross a blocked diagonal corner");
}

void TestInvalidGridRejected() {
    AStarSolver solver;
    Require(!solver.InitGrid(3, 3, std::vector<bool>(8, false)),
            "bitmap shape mismatch must be rejected");
}

}  // namespace

int main() {
    TestUnitCostShortestPath();
    TestCornerCutRejected();
    TestInvalidGridRejected();
    std::cout << "astar_solver_contract: PASS" << std::endl;
    return 0;
}
