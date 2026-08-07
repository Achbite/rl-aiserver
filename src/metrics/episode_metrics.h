#pragma once

#include "contracts/contract_namespaces.h"

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct SingleMapTaskSnapshot;

struct AgentEpisodeResult {
    double episode_return = 0.0;
    bool success = false;
    maze::MazeTerminationReason termination_reason =
        maze::MAZE_TERMINATION_REASON_UNSPECIFIED;
    int64_t transition_count = 0;
    int64_t shortest_action_steps = 0;
    int64_t unique_cell_count = 0;
    int64_t blocked_move_count = 0;
    std::unordered_map<std::string, double> reward_component_sums;
};

class EpisodeMetricsWindow {
public:
    explicit EpisodeMetricsWindow(std::size_t capacity);

    void AddCompleted(std::vector<AgentEpisodeResult> agents);
    void AddExcluded(std::size_t agent_count,
                     maze::MazeTerminationReason reason);
    void Fill(training::MetricSnapshot* snapshot,
              const common::ServiceInstanceIdentity& source,
              uint64_t sequence,
              int64_t timestamp_unix_ms) const;

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

// Appends task-control metrics to the same AIServer snapshot as episode
// metrics. Values that do not yet exist (for example, evaluation results
// before the first completed campaign) are deliberately omitted.
void AppendSingleMapTaskMetrics(
    training::MetricSnapshot* snapshot,
    const SingleMapTaskSnapshot& task,
    bool has_completed_evaluation,
    int64_t timestamp_unix_ms);
