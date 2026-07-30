#include "metrics/episode_metrics.h"

#include <algorithm>
#include <limits>
#include <map>
#include <utility>

EpisodeMetricsWindow::EpisodeMetricsWindow(std::size_t capacity)
    : capacity_(std::max<std::size_t>(1, capacity)) {}

void EpisodeMetricsWindow::Push(Entry entry) {
    entries_.push_back(std::move(entry));
    while (entries_.size() > capacity_) {
        entries_.pop_front();
    }
}

void EpisodeMetricsWindow::AddCompleted(
    std::vector<AgentEpisodeResult> agents) {
    std::lock_guard<std::mutex> lock(mutex_);
    Push(Entry{false, std::move(agents)});
}

void EpisodeMetricsWindow::AddExcluded(
    std::size_t agent_count,
    maze::TerminationReason reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<AgentEpisodeResult> agents(agent_count);
    for (auto& agent : agents) {
        agent.termination_reason = reason;
    }
    Push(Entry{true, std::move(agents)});
}

void EpisodeMetricsWindow::Fill(maze::EpisodeMetrics* metrics) const {
    std::lock_guard<std::mutex> lock(mutex_);
    metrics->Clear();
    metrics->set_configured_window_size(static_cast<int32_t>(capacity_));

    int64_t completed_episodes = 0;
    int64_t excluded_episodes = 0;
    int64_t completed_agents = 0;
    int64_t agent_successes = 0;
    int64_t any_successes = 0;
    int64_t all_successes = 0;
    int64_t reward_transitions = 0;
    double return_sum = 0.0;
    double return_min = std::numeric_limits<double>::infinity();
    double return_max = -std::numeric_limits<double>::infinity();
    std::map<maze::TerminationReason, int64_t> termination_counts;
    std::unordered_map<std::string, double> reward_component_sums;

    for (const auto& entry : entries_) {
        if (entry.excluded) {
            ++excluded_episodes;
            for (const auto& agent : entry.agents) {
                ++termination_counts[agent.termination_reason];
            }
            continue;
        }

        ++completed_episodes;
        bool any_success = false;
        bool all_success = !entry.agents.empty();
        for (const auto& agent : entry.agents) {
            ++completed_agents;
            return_sum += agent.episode_return;
            return_min = std::min(return_min, agent.episode_return);
            return_max = std::max(return_max, agent.episode_return);
            ++termination_counts[agent.termination_reason];
            if (agent.success) {
                ++agent_successes;
                any_success = true;
            } else {
                all_success = false;
            }
            reward_transitions += agent.transition_count;
            for (const auto& item : agent.reward_component_sums) {
                reward_component_sums[item.first] += item.second;
            }
        }
        if (any_success) ++any_successes;
        if (all_success) ++all_successes;
    }

    metrics->set_completed_episode_count(completed_episodes);
    metrics->set_excluded_episode_count(excluded_episodes);
    metrics->set_completed_agent_count(completed_agents);
    metrics->set_agent_success_count(agent_successes);
    metrics->set_environment_any_success_count(any_successes);
    metrics->set_environment_all_success_count(all_successes);
    if (completed_agents > 0) {
        metrics->set_mean_agent_return(return_sum / completed_agents);
        metrics->set_min_agent_return(return_min);
        metrics->set_max_agent_return(return_max);
        metrics->set_agent_success_rate(
            static_cast<double>(agent_successes) / completed_agents);
    }
    if (completed_episodes > 0) {
        metrics->set_environment_any_success_rate(
            static_cast<double>(any_successes) / completed_episodes);
        metrics->set_environment_all_success_rate(
            static_cast<double>(all_successes) / completed_episodes);
    }
    for (const auto& item : termination_counts) {
        auto* count = metrics->add_termination_counts();
        count->set_reason(item.first);
        count->set_count(item.second);
    }
    if (reward_transitions > 0) {
        for (const auto& item : reward_component_sums) {
            (*metrics->mutable_reward_component_mean())[item.first] =
                item.second / reward_transitions;
        }
    }
}
