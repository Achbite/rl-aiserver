#include "ai/maze_reward.h"

#include <cmath>
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

bool Near(float left, float right, float tolerance = 1e-6f) {
    return std::abs(left - right) <= tolerance;
}

float Component(const RewardDetail& detail, const std::string& name) {
    for (const auto& item : detail.items) {
        if (item.first == name) return item.second;
    }
    std::cerr << "FAIL: missing Reward component " << name << std::endl;
    std::exit(1);
}

void RequireAtomicSum(const RewardDetail& detail) {
    Require(detail.valid, "Reward must be valid: " + detail.error);
    float sum = 0.0f;
    for (const auto& item : detail.items) {
        Require(std::isfinite(item.second),
                "Reward component must be finite: " + item.first);
        sum += item.second;
    }
    Require(detail.items.size() == 5,
            "Reward V3 exposes exactly five atomic components");
    Require(Near(sum, detail.total),
            "PPO Reward equals the atomic component sum");
    Require(Near(detail.task_total + detail.shaping_total, detail.total),
            "task and shaping partitions equal PPO Reward");
}

MazeRewardConfig RewardConfig() {
    MazeRewardConfig config;
    config.goal_reward = 10.0f;
    config.timeout_base = -2.0f;
    config.gamma = 0.99f;
    config.potential_distance_scale = 220;
    config.stage_8x_first_visit_cap = 0.25f;
    config.stage_4x_first_visit_cap = 0.10f;
    config.stage_2x_first_visit_cap = 0.0f;
    return config;
}

SessionManager::Session MakeSession() {
    SessionManager::Session session;
    session.grid_cols = 221;
    session.grid_rows = 1;
    session.start_gx = 188;
    session.start_gy = 0;
    session.end_gx = 0;
    session.end_gy = 0;
    session.shortest_action_steps = 188;
    session.max_finite_geodesic_distance = 220;
    session.current_max_steps = 1504;
    session.curriculum_stage = maze::CURRICULUM_STAGE_8X;
    session.blocked = std::vector<bool>(221);
    for (int index = 0; index <= 220; ++index) {
        session.geodesic_distance.push_back(index);
    }
    session.agents.emplace(1, SessionManager::AgentRuntime{});
    session.agents.at(1).visited.insert(188);
    return session;
}

RewardDetail Transition(
    SessionManager::Session& session,
    int from_x,
    int to_x,
    bool is_done = false,
    maze::MazeTerminationReason reason =
        maze::MAZE_TERMINATION_REASON_ACTIVE) {
    auto& agent = session.agents.at(1);
    agent.prev_grid_x = from_x;
    agent.prev_grid_y = 0;
    const bool moved = from_x != to_x;
    agent.current_state_first_visit =
        moved && agent.visited.insert(to_x).second;
    return MazeReward::Calculate(
        session, 1, to_x, 0, is_done, reason, RewardConfig());
}

void TestGeodesicPbrsAndExplorationTolerance() {
    auto session = MakeSession();
    const float first_visit_unit = 0.25f / 188.0f;

    const auto progress = Transition(session, 188, 187);
    RequireAtomicSum(progress);
    Require(Component(progress, "geodesic_pbrs") > 0.0f,
            "geodesic progress produces positive PBRS shaping");
    Require(Near(Component(progress, "first_visit_bonus"),
                 first_visit_unit),
            "8x first visit uses cap divided by d-star");

    const auto retreat = Transition(session, 187, 188);
    RequireAtomicSum(retreat);
    Require(Component(retreat, "geodesic_pbrs") < 0.0f,
            "moving away from Goal produces negative PBRS shaping");
    Require(Near(Component(retreat, "first_visit_bonus"), 0.0f),
            "revisited cells do not receive exploration credit");

    const auto stay = Transition(session, 100, 100);
    RequireAtomicSum(stay);
    Require(Component(stay, "geodesic_pbrs") <= 0.0f,
            "Stay and blocked moves cannot farm positive shaping");

    const float closed_loop =
        Component(Transition(session, 188, 187), "geodesic_pbrs") +
        Component(Transition(session, 187, 188), "geodesic_pbrs");
    Require(closed_loop <= 0.0f,
            "a closed loop cannot accumulate positive PBRS shaping");
}

void TestFirstVisitStagesAndCaps() {
    {
        auto session = MakeSession();
        auto& agent = session.agents.at(1);
        const float unit_8x = 0.25f / 188.0f;
        agent.first_visit_bonus_total = 0.25f - unit_8x / 2.0f;
        const auto capped = Transition(session, 188, 187);
        Require(Near(Component(capped, "first_visit_bonus"),
                     unit_8x / 2.0f),
                "8x first-visit total is capped at 0.25");
    }
    {
        auto session = MakeSession();
        session.curriculum_stage = maze::CURRICULUM_STAGE_4X;
        auto& agent = session.agents.at(1);
        const float unit_4x = 0.10f / 188.0f;
        agent.first_visit_bonus_total = 0.10f - unit_4x / 2.0f;
        const auto capped = Transition(session, 188, 187);
        Require(Near(Component(capped, "first_visit_bonus"),
                     unit_4x / 2.0f),
                "4x first-visit total is capped at 0.10");
    }
    {
        auto session = MakeSession();
        session.curriculum_stage = maze::CURRICULUM_STAGE_2X;
        const auto final_stage = Transition(session, 188, 187);
        Require(Near(Component(final_stage, "first_visit_bonus"), 0.0f),
                "2x final stage disables first-visit bonus");
    }
}

void TestGoalAndTimeoutTerminalTerms() {
    auto session = MakeSession();
    const auto goal = Transition(
        session, 1, 0, true,
        maze::MAZE_TERMINATION_REASON_GOAL_REACHED);
    RequireAtomicSum(goal);
    Require(Near(Component(goal, "goal_reward"), 10.0f),
            "Goal receives +10");
    Require(Near(Component(goal, "timeout_base"), 0.0f) &&
                Near(Component(goal, "timeout_progress"), 0.0f),
            "Goal never receives TIME_LIMIT terms");
    Require(goal.total > 0.0f,
            "Goal terminal transition remains positive");

    const auto timeout = Transition(
        session, 219, 220, true,
        maze::MAZE_TERMINATION_REASON_TIME_LIMIT);
    RequireAtomicSum(timeout);
    Require(Near(Component(timeout, "goal_reward"), 0.0f),
            "TIME_LIMIT never receives Goal Reward");
    Require(Near(Component(timeout, "timeout_base"), -2.0f),
            "TIME_LIMIT receives the fixed base penalty");
    Require(Near(Component(timeout, "timeout_progress"),
                 (188.0f - 220.0f) / 188.0f),
            "TIME_LIMIT progress uses (d-star minus d-terminal) over d-star");
    Require(timeout.total < 0.0f,
            "a failed terminal transition is negative");
}

void TestFailureReturnAndFailClosedStates() {
    auto session = MakeSession();
    float episode_return = 0.0f;
    for (int step = 0; step < 8; ++step) {
        const bool terminal = step == 7;
        auto detail = Transition(
            session, 188, 188, terminal,
            terminal ? maze::MAZE_TERMINATION_REASON_TIME_LIMIT
                     : maze::MAZE_TERMINATION_REASON_ACTIVE);
        RequireAtomicSum(detail);
        episode_return += detail.total;
    }
    Require(episode_return < 0.0f,
            "Episode without success has negative learning return");

    session.geodesic_distance[187] = -1;
    Require(!Transition(session, 188, 187).valid,
            "unreachable runtime state fails closed");
    Require(!Transition(
                 session, 1, 1, true,
                 maze::MAZE_TERMINATION_REASON_GOAL_REACHED)
                 .valid,
            "Goal reason outside Goal cell fails closed");
}

}  // namespace

int main() {
    TestGeodesicPbrsAndExplorationTolerance();
    TestFirstVisitStagesAndCaps();
    TestGoalAndTimeoutTerminalTerms();
    TestFailureReturnAndFailClosedStates();
    std::cout << "reward_contract: PASS\n";
    return 0;
}
