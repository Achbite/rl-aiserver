#pragma once

#include "contracts/contract_namespaces.h"
#include "session/session_manager.h"
#include "session/training_session.h"

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

struct MazeAgentState {
    int prev_grid_x = -1;
    int prev_grid_y = -1;
    bool reached_goal = false;
    std::unordered_set<int> visited;
    bool current_state_first_visit = false;
    float first_visit_bonus_total = 0.0f;
    int episode_start_geodesic_distance = -1;
    int observation_grid_x = -1;
    int observation_grid_y = -1;
    bool last_move_blocked = false;
    int64_t blocked_move_count = 0;
    maze::MazeTerminationReason final_termination_reason =
        maze::MAZE_TERMINATION_REASON_UNSPECIFIED;
};

struct MazeSessionState {
    std::string map_id;
    int shortest_action_steps = 0;
    float map_width = 0.0f;
    float map_height = 0.0f;
    float start_x = 0.0f;
    float start_y = 0.0f;
    float end_x = 0.0f;
    float end_y = 0.0f;
    int start_gx = 0;
    int start_gy = 0;
    int end_gx = 0;
    int end_gy = 0;
    int grid_cols = 0;
    int grid_rows = 0;
    uint32_t grid_size_microunits = 0;
    std::vector<bool> blocked;
    std::vector<int> geodesic_distance;
    int max_finite_geodesic_distance = -1;
    int current_max_steps = 0;

    bool IsWalkable(int gx, int gy) const {
        if (gx < 0 || gx >= grid_cols || gy < 0 || gy >= grid_rows) {
            return false;
        }
        return !blocked[gy * grid_cols + gx];
    }
};

struct MazeAgentRuntime : AgentTrainingState {
    MazeAgentState task;
};

struct MazeSession : TrainingSession<MazeAgentRuntime> {
    MazeSessionState task;
};

using MazeSessionManager = SessionManager<MazeSession>;
