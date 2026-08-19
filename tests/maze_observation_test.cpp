#include "ai/maze_observation.h"

#include <cmath>
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

bool Near(float left, float right, float tolerance = 1e-6f) {
    return std::abs(left - right) <= tolerance;
}

SessionManager::Session MakeSession() {
    SessionManager::Session session;
    session.grid_cols = 3;
    session.grid_rows = 3;
    session.start_gx = 0;
    session.start_gy = 0;
    session.end_gx = 2;
    session.end_gy = 2;
    session.current_max_steps = 8;
    session.blocked.assign(9, false);
    return session;
}

SessionManager::AgentRuntime MakeAgent() {
    SessionManager::AgentRuntime agent;
    agent.visited.insert(0);
    agent.observation_grid_x = 0;
    agent.observation_grid_y = 0;
    return agent;
}

void TestSeventeenDimensionContract() {
    auto session = MakeSession();
    auto agent = MakeAgent();
    std::vector<float> observation;
    std::string error;
    Require(MazeObservation::Build(
                session, agent, 0, 0, 0, 10, 17,
                observation, error),
            "initial observation must build: " + error);
    Require(observation.size() == 17,
            "maze.observation.v3 has exactly 17 values");
    Require(Near(observation[0], 0.0f) && Near(observation[1], 0.0f),
            "Agent coordinates occupy indices 0 and 1");
    Require(Near(observation[2], 1.0f) && Near(observation[3], 1.0f),
            "Goal coordinates occupy indices 2 and 3");
    Require(Near(observation[4], std::sqrt(0.5f)) &&
                Near(observation[5], std::sqrt(0.5f)),
            "bearing unit vector occupies indices 4 and 5");
    Require(Near(observation[6], 1.0f),
            "Euclidean Goal distance is normalized at index 6");
    Require(Near(observation[15], 1.0f),
            "remaining step ratio starts at one");
    Require(Near(observation[16], 0.0f),
            "last_move_blocked starts at zero");
    for (float value : observation) {
        Require(std::isfinite(value), "observation values are finite");
    }

    Require(!MazeObservation::Build(
                session, agent, 0, 0, 0, 10, 13,
                observation, error),
            "legacy 13-dimensional model fails closed");
    Require(!MazeObservation::Build(
                session, agent, 0, 0, 0, 0, 17,
                observation, error),
            "invalid ray range fails closed");
}

void TestStateTransitionsAndBlockedSignal() {
    auto session = MakeSession();
    auto agent = MakeAgent();
    std::string error;
    Require(MazeObservation::ApplyState(
                session, agent, 0, 0, 0, false, false, error),
            "frame zero accepts the assigned start");
    Require(agent.visited.size() == 1 &&
                !agent.current_state_first_visit,
            "frame zero does not grant first-visit credit");

    agent.last_action = 3;
    Require(MazeObservation::ApplyState(
                session, agent, 1, 0, 1, false, false, error),
            "successful movement applies");
    Require(agent.current_state_first_visit && !agent.last_move_blocked,
            "new cell and unblocked action are recorded");

    agent.last_action = 3;
    Require(!MazeObservation::ApplyState(
                session, agent, 1, 0, 2, false, false, error),
            "Client cannot hide a blocked non-Stay action");
    Require(MazeObservation::ApplyState(
                session, agent, 1, 0, 2, false, true, error),
            "matching blocked signal applies");
    Require(agent.last_move_blocked && agent.blocked_move_count == 1,
            "blocked move is tracked only by AIServer state");

    agent.last_action = 7;
    Require(MazeObservation::ApplyState(
                session, agent, 0, 0, 3, true, false, error),
            "terminal transition applies");
    Require(!MazeObservation::ApplyState(
                session, agent, 0, 0, 4, false, false, error),
            "terminal state cannot become active again");
}

void TestFrameAndStartIdentity() {
    const auto session = MakeSession();
    auto agent = MakeAgent();
    std::string error;
    Require(!MazeObservation::ApplyState(
                session, agent, 1, 0, 0, false, false, error),
            "frame zero outside assigned start fails");
    agent = MakeAgent();
    Require(MazeObservation::ApplyState(
                session, agent, 0, 0, 0, false, false, error),
            "valid start initializes observation state");
    Require(!MazeObservation::ApplyState(
                session, agent, 1, 0, 2, false, false, error),
            "non-contiguous observation frame fails");
}

}  // namespace

int main() {
    TestSeventeenDimensionContract();
    TestStateTransitionsAndBlockedSignal();
    TestFrameAndStartIdentity();
    std::cout << "maze_observation_contract: PASS\n";
    return 0;
}
