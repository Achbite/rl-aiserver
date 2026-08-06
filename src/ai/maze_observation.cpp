#include "ai/maze_observation.h"

#include <algorithm>
#include <cmath>

bool MazeObservation::Build(
    const SessionManager::Session& session,
    const SessionManager::AgentRuntime& agent,
    int gx,
    int gy,
    int64_t episode_step,
    int ray_max_range,
    int expected_obs_dim,
    std::vector<float>& observation,
    std::string& error) {
    observation.assign(static_cast<std::size_t>(expected_obs_dim), 0.0f);
    if (expected_obs_dim != 17 || ray_max_range <= 0 ||
        session.current_max_steps <= 0 || episode_step < 0 ||
        !session.IsWalkable(gx, gy)) {
        error = "observation dimensions, step, or grid state are invalid";
        return false;
    }

    observation[0] = session.grid_cols > 1
                         ? static_cast<float>(gx) / (session.grid_cols - 1)
                         : 0.0f;
    observation[1] = session.grid_rows > 1
                         ? static_cast<float>(gy) / (session.grid_rows - 1)
                         : 0.0f;
    observation[2] = session.grid_cols > 1
                         ? static_cast<float>(session.end_gx) /
                               (session.grid_cols - 1)
                         : 0.0f;
    observation[3] = session.grid_rows > 1
                         ? static_cast<float>(session.end_gy) /
                               (session.grid_rows - 1)
                         : 0.0f;
    const float dx = static_cast<float>(session.end_gx - gx);
    const float dy = static_cast<float>(session.end_gy - gy);
    const float distance = std::sqrt(dx * dx + dy * dy);
    if (distance > 0.0f) {
        observation[4] = dx / distance;
        observation[5] = dy / distance;
    }
    const float max_distance = std::sqrt(static_cast<float>(
        (session.grid_cols - 1) * (session.grid_cols - 1) +
        (session.grid_rows - 1) * (session.grid_rows - 1)));
    observation[6] =
        max_distance > 0.0f ? distance / max_distance : 0.0f;
    static constexpr int directions[8][2] = {
        {0, 1}, {1, 1}, {1, 0}, {1, -1},
        {0, -1}, {-1, -1}, {-1, 0}, {-1, 1},
    };
    for (int index = 0; index < 8; ++index) {
        const int dx_ray = directions[index][0];
        const int dy_ray = directions[index][1];
        int open_steps = 0;
        int previous_x = gx;
        int previous_y = gy;
        for (int step = 1; step <= ray_max_range; ++step) {
            const int next_x = gx + dx_ray * step;
            const int next_y = gy + dy_ray * step;
            if (!session.IsWalkable(next_x, next_y)) break;
            if (dx_ray != 0 && dy_ray != 0 &&
                (!session.IsWalkable(previous_x + dx_ray, previous_y) ||
                 !session.IsWalkable(previous_x, previous_y + dy_ray))) {
                break;
            }
            open_steps = step;
            previous_x = next_x;
            previous_y = next_y;
        }
        observation[7 + index] =
            static_cast<float>(open_steps) / ray_max_range;
    }

    const double horizon = static_cast<double>(session.current_max_steps);
    observation[15] = static_cast<float>(
        std::max(0.0, horizon - static_cast<double>(episode_step)) /
        horizon);
    observation[16] = agent.last_move_blocked ? 1.0f : 0.0f;

    if (!std::all_of(
            observation.begin(), observation.end(),
            [](float value) { return std::isfinite(value); })) {
        error = "observation contains a non-finite value";
        return false;
    }
    error.clear();
    return true;
}

bool MazeObservation::ApplyState(
    const SessionManager::Session& session,
    SessionManager::AgentRuntime& agent,
    int gx,
    int gy,
    int64_t frame_id,
    bool is_done,
    bool reported_last_move_blocked,
    std::string& error) {
    if (!session.IsWalkable(gx, gy)) {
        error = "observation state is not walkable";
        return false;
    }
    if (agent.observation_done) {
        if (!is_done) {
            error = "terminal observation became active";
            return false;
        }
        return true;
    }
    if (frame_id == 0 && agent.last_observation_frame_id < 0) {
        if (gx != session.start_gx || gy != session.start_gy) {
            error = "initial observation is not at the assigned start";
            return false;
        }
        if (reported_last_move_blocked) {
            error = "initial observation cannot report a blocked move";
            return false;
        }
        agent.last_observation_frame_id = 0;
        agent.observation_done = is_done;
        error.clear();
        return true;
    }
    if (agent.last_observation_frame_id < 0 ||
        frame_id != agent.last_observation_frame_id + 1) {
        error = "observation frame is not contiguous";
        return false;
    }

    const bool moved =
        gx != agent.observation_grid_x || gy != agent.observation_grid_y;
    const bool expected_last_move_blocked = agent.last_action != 0 && !moved;
    if (reported_last_move_blocked != expected_last_move_blocked) {
        error = "Client last_move_blocked does not match the executed action";
        return false;
    }
    agent.last_move_blocked = expected_last_move_blocked;
    if (agent.last_move_blocked) {
        ++agent.blocked_move_count;
    }
    const int key = gy * session.grid_cols + gx;
    agent.current_state_first_visit = moved && agent.visited.insert(key).second;
    agent.observation_grid_x = gx;
    agent.observation_grid_y = gy;
    agent.last_observation_frame_id = frame_id;
    agent.observation_done = is_done;
    error.clear();
    return true;
}
