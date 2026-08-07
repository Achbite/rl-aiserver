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
            "Reward V4 exposes exactly five atomic components");
    Require(Near(sum, detail.total),
            "PPO Reward equals the atomic component sum");
    Require(Near(detail.task_total + detail.shaping_total, detail.total),
            "task and shaping partitions equal PPO Reward");
}

MazeRewardConfig RewardConfig() {
    MazeRewardConfig config;
    config.goal_reward = 10.0f;
    config.timeout_penalty = -2.0f;
    config.progress_budget = 1.0f;
    config.stage_8x_first_visit_budget = 0.75f;
    config.stage_4x_first_visit_budget = 0.25f;
    config.stage_2x_first_visit_budget = 0.0f;
    config.wasted_action_penalty = -0.002f;
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
    auto& agent = session.agents.at(1);
    agent.visited.insert(188);
    agent.episode_start_geodesic_distance = 188;
    return session;
}

RewardDetail Transition(
    SessionManager::Session& session,
    int from_x,
    int to_x,
    bool is_done = false,
    maze::MazeTerminationReason reason =
        maze::MAZE_TERMINATION_REASON_ACTIVE,
    const MazeRewardConfig& config = RewardConfig()) {
    auto& agent = session.agents.at(1);
    agent.prev_grid_x = from_x;
    agent.prev_grid_y = 0;
    const bool moved = from_x != to_x;
    agent.current_state_first_visit =
        moved && agent.visited.insert(to_x).second;
    RewardDetail detail = MazeReward::Calculate(
        session, 1, to_x, 0, is_done, reason, config);
    if (detail.valid) {
        agent.first_visit_bonus_total +=
            Component(detail, "first_visit_bonus");
    }
    return detail;
}

void TestProgressHasNoDistanceDependentSignFlip() {
    auto session = MakeSession();
    const float expected = 1.0f / 188.0f;
    for (int distance = 188; distance > 0; --distance) {
        const bool goal = distance == 1;
        const auto detail = Transition(
            session, distance, distance - 1, goal,
            goal ? maze::MAZE_TERMINATION_REASON_GOAL_REACHED
                 : maze::MAZE_TERMINATION_REASON_ACTIVE);
        RequireAtomicSum(detail);
        Require(Near(Component(detail, "geodesic_progress"), expected),
                "every one-step approach from d=188 through d=1 has the same positive progress credit");
    }

    session = MakeSession();
    for (int distance = 1; distance < 188; ++distance) {
        const auto detail = Transition(session, distance, distance + 1);
        RequireAtomicSum(detail);
        Require(Near(Component(detail, "geodesic_progress"), -expected),
                "every one-step retreat has the same negative progress credit");
    }
}

void TestLoopsDetoursAndWastedActions() {
    auto session = MakeSession();
    const float forward = Component(
        Transition(session, 188, 187), "geodesic_progress");
    const float backward = Component(
        Transition(session, 187, 188), "geodesic_progress");
    Require(Near(forward + backward, 0.0f),
            "geodesic progress telescopes to zero over a closed loop");

    session = MakeSession();
    auto& agent = session.agents.at(1);
    agent.visited.clear();
    agent.visited.insert(187);
    const auto new_cell_detour = Transition(session, 187, 188);
    Require(Component(new_cell_detour, "geodesic_progress") < 0.0f,
            "moving away remains negative progress");
    Require(Component(new_cell_detour, "first_visit_bonus") > 0.0f,
            "a new detour cell receives bounded exploration credit");
    Require(std::abs(new_cell_detour.shaping_total) <
                std::abs(Component(new_cell_detour, "geodesic_progress")),
            "first visit softens but does not reverse a one-step detour");

    const auto revisit = Transition(session, 188, 187);
    Require(Near(Component(revisit, "first_visit_bonus"), 0.0f),
            "revisited cells receive no exploration credit");

    const auto stay = Transition(session, 100, 100);
    Require(Near(Component(stay, "geodesic_progress"), 0.0f),
            "unchanged distance has zero progress credit");
    Require(Near(Component(stay, "wasted_action_penalty"), -0.002f),
            "Stay or collision receives the configured wasted-action penalty");
    Require(stay.total < 0.0f,
            "an unchanged action cannot farm positive Reward");
}

void TestFirstVisitStagesAndBudgets() {
    {
        auto session = MakeSession();
        auto& agent = session.agents.at(1);
        const float unit = 0.75f / 188.0f;
        agent.first_visit_bonus_total = 0.75f - unit / 2.0f;
        const auto capped = Transition(session, 188, 187);
        Require(Near(Component(capped, "first_visit_bonus"), unit / 2.0f),
                "8x first-visit total is capped at 0.75");
    }
    {
        auto session = MakeSession();
        session.curriculum_stage = maze::CURRICULUM_STAGE_4X;
        auto& agent = session.agents.at(1);
        const float unit = 0.25f / 188.0f;
        agent.first_visit_bonus_total = 0.25f - unit / 2.0f;
        const auto capped = Transition(session, 188, 187);
        Require(Near(Component(capped, "first_visit_bonus"), unit / 2.0f),
                "4x first-visit total is capped at 0.25");
    }
    {
        auto session = MakeSession();
        session.curriculum_stage = maze::CURRICULUM_STAGE_2X;
        const auto final_stage = Transition(session, 188, 187);
        Require(Near(Component(final_stage, "first_visit_bonus"), 0.0f),
                "2x final stage disables first-visit bonus");
    }
}

void TestGoalTimeoutAndFailureReturn() {
    auto session = MakeSession();
    const auto goal = Transition(
        session, 1, 0, true,
        maze::MAZE_TERMINATION_REASON_GOAL_REACHED);
    RequireAtomicSum(goal);
    Require(Near(Component(goal, "goal_reward"), 10.0f),
            "Goal receives +10");
    Require(Near(Component(goal, "timeout_penalty"), 0.0f),
            "Goal never receives TIME_LIMIT penalty");
    Require(goal.total > 0.0f,
            "Goal terminal transition remains positive");

    session = MakeSession();
    float failure_return = 0.0f;
    for (int distance = 188; distance > 1; --distance) {
        const auto detail = Transition(session, distance, distance - 1);
        RequireAtomicSum(detail);
        failure_return += detail.total;
    }
    const auto timeout = Transition(
        session, 1, 1, true,
        maze::MAZE_TERMINATION_REASON_TIME_LIMIT);
    RequireAtomicSum(timeout);
    failure_return += timeout.total;
    Require(Near(Component(timeout, "goal_reward"), 0.0f),
            "TIME_LIMIT never receives Goal Reward");
    Require(Near(Component(timeout, "timeout_penalty"), -2.0f),
            "TIME_LIMIT receives the fixed penalty");
    Require(failure_return < 0.0f,
            "even the maximum-progress 8x failure Episode remains strictly negative");
}

void TestFailClosedStatesAndConfiguration() {
    auto session = MakeSession();
    session.geodesic_distance[186] = 186;
    session.geodesic_distance[187] = 188;
    Require(!Transition(session, 187, 186).valid,
            "geodesic delta greater than one fails closed");

    session = MakeSession();
    session.geodesic_distance[187] = -1;
    Require(!Transition(session, 188, 187).valid,
            "unreachable runtime state fails closed");
    Require(!Transition(
                 session, 1, 1, true,
                 maze::MAZE_TERMINATION_REASON_GOAL_REACHED)
                 .valid,
            "Goal reason outside Goal cell fails closed");

    session = MakeSession();
    auto unsafe = RewardConfig();
    unsafe.timeout_penalty = -1.0f;
    Require(!Transition(session, 188, 187, false,
                        maze::MAZE_TERMINATION_REASON_ACTIVE, unsafe)
                 .valid,
            "non-negative worst-case failure budget is rejected");
}

}  // namespace

int main() {
    TestProgressHasNoDistanceDependentSignFlip();
    TestLoopsDetoursAndWastedActions();
    TestFirstVisitStagesAndBudgets();
    TestGoalTimeoutAndFailureReturn();
    TestFailClosedStatesAndConfiguration();
    std::cout << "reward_contract: PASS\n";
    return 0;
}
