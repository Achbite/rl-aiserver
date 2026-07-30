#include "metrics/episode_metrics.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << std::endl;
        std::exit(1);
    }
}

bool Near(double left, double right) {
    return std::abs(left - right) < 1e-9;
}

}  // namespace

int main() {
    EpisodeMetricsWindow window(2);
    AgentEpisodeResult success;
    success.episode_return = 12.0;
    success.success = true;
    success.termination_reason = maze::TERMINATION_REASON_GOAL_REACHED;
    success.transition_count = 2;
    success.reward_component_sums["goal"] = 10.0;

    AgentEpisodeResult timeout;
    timeout.episode_return = -2.0;
    timeout.termination_reason = maze::TERMINATION_REASON_TIME_LIMIT;
    timeout.transition_count = 2;
    timeout.reward_component_sums["goal"] = 0.0;

    window.AddCompleted({success, timeout});
    window.AddExcluded(2, maze::TERMINATION_REASON_CHAIN_FAILURE);

    maze::EpisodeMetrics metrics;
    window.Fill(&metrics);
    Require(metrics.configured_window_size() == 2, "window size");
    Require(metrics.completed_episode_count() == 1, "completed episode");
    Require(metrics.excluded_episode_count() == 1, "excluded episode");
    Require(metrics.completed_agent_count() == 2, "completed agents");
    Require(Near(metrics.mean_agent_return(), 5.0), "mean return");
    Require(Near(metrics.agent_success_rate(), 0.5), "agent success rate");
    Require(Near(metrics.environment_any_success_rate(), 1.0),
            "any success rate");
    Require(Near(metrics.environment_all_success_rate(), 0.0),
            "all success rate");
    Require(Near(metrics.reward_component_mean().at("goal"), 2.5),
            "reward component mean");
    return 0;
}
