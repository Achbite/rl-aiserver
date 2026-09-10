#include "maze/protocol/adapter.h"

#include "maze/observation/observation.h"

MazeTaskAdapter::MazeTaskAdapter(const MazeConfig& config)
    : config_(config) {}

void MazeTaskAdapter::RegisterMetrics(MetricRegistry& registry) const {
    MazeReward::RegisterMetrics(registry);
    RegisterMazeEpisodeMetrics(registry);
}

void MazeTaskAdapter::ResetEpisodeState(
    MazeSessionManager::Session& session, const std::string& episode_id) const {
    session.ResetForEpisode(episode_id);
    for (auto& item : session.agents) {
        auto& agent = item.second;
        agent.task.prev_grid_x = -1;
        agent.task.prev_grid_y = -1;
        agent.task.reached_goal = false;
        agent.task.visited.clear();
        agent.task.current_state_first_visit = false;
        agent.task.first_visit_bonus_total = 0.0f;
        agent.task.episode_start_geodesic_distance = -1;
        agent.task.observation_grid_x = -1;
        agent.task.observation_grid_y = -1;
        agent.task.last_move_blocked = false;
        agent.task.blocked_move_count = 0;
        agent.task.final_termination_reason =
            maze::MAZE_TERMINATION_REASON_UNSPECIFIED;
    }
}

training::RegisteredMetricRecord MazeTaskAdapter::BuildEpisodeMetricFact(
    MetricRegistry& registry, const MazeSessionManager::Session& session,
    const std::vector<AgentEpisodeResult>& agents) const {
    return BuildMazeEpisodeMetrics(registry, session.environment_instance_id,
                                   session.current_episode_id, agents);
}

maze::EpisodeOutcome MazeTaskAdapter::BuildEpisodeOutcome(
    const MazeSessionManager::Session& session,
    const std::vector<AgentEpisodeResult>& agents) const {
    maze::EpisodeOutcome outcome;
    outcome.set_episode_id(session.current_episode_id);
    for (const auto& agent : agents) {
        auto* target = outcome.add_agents();
        target->set_agent_id(agent.agent_id);
        target->set_termination_reason(agent.termination_reason);
        target->set_terminal_frame_id(agent.terminal_frame_id);
        target->mutable_final_position()->set_x(
            static_cast<float>(agent.final_grid_x));
        target->mutable_final_position()->set_y(
            static_cast<float>(agent.final_grid_y));
        if (agent.goal_rank_group) {
            target->set_goal_rank_group(*agent.goal_rank_group);
        }
    }
    return outcome;
}

bool MazeTaskAdapter::BuildObservation(
    const MazeSessionManager::Session& session,
    const MazeSessionManager::AgentRuntime& agent,
    int gx, int gy, int64_t frame_id,
    std::vector<float>& observation, std::string& error) const {
    return MazeObservation::Build(
        session, agent, gx, gy, frame_id,
        config_.observation.ray_max_range, config_.model.expected_obs_dim,
        observation, error);
}

MazeRewardDetail MazeTaskAdapter::CalculateReward(
    const MazeSessionManager::Session& session,
    const MazeSessionManager::AgentRuntime& agent,
    int gx, int gy, bool is_done,
    maze::MazeTerminationReason reason) const {
    return MazeReward::Calculate(
        {session.task.grid_cols, session.task.grid_rows, session.task.geodesic_distance,
         session.task.shortest_action_steps, agent.task.prev_grid_x, agent.task.prev_grid_y,
         agent.task.episode_start_geodesic_distance, agent.task.current_state_first_visit,
         agent.task.first_visit_bonus_total}, gx, gy, is_done, reason);
}
