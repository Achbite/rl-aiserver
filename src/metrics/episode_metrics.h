#pragma once

#include "maze.pb.h"

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct AgentEpisodeResult {
    double episode_return = 0.0;
    bool success = false;
    maze::TerminationReason termination_reason =
        maze::TERMINATION_REASON_UNSPECIFIED;
    int64_t transition_count = 0;
    std::unordered_map<std::string, double> reward_component_sums;
};

class EpisodeMetricsWindow {
public:
    explicit EpisodeMetricsWindow(std::size_t capacity);

    void AddCompleted(std::vector<AgentEpisodeResult> agents);
    void AddExcluded(std::size_t agent_count,
                     maze::TerminationReason reason);
    void Fill(maze::EpisodeMetrics* metrics) const;

private:
    struct Entry {
        bool excluded = false;
        std::vector<AgentEpisodeResult> agents;
    };

    void Push(Entry entry);

    std::size_t capacity_;
    mutable std::mutex mutex_;
    std::deque<Entry> entries_;
};
