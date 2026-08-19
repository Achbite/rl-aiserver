#pragma once

#include "contracts/contract_namespaces.h"

#include <cstdint>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct AgentEpisodeResult {
    uint32_t agent_id = 0;
    double episode_return = 0.0;
    bool success = false;
    maze::MazeTerminationReason termination_reason =
        maze::MAZE_TERMINATION_REASON_UNSPECIFIED;
    int64_t transition_count = 0;
    int64_t shortest_action_steps = 0;
    int64_t unique_cell_count = 0;
    int64_t blocked_move_count = 0;
    int64_t attempted_move_count = 0;
    uint64_t minimum_behavior_model_step = 0;
    uint64_t maximum_behavior_model_step = 0;
    std::string behavior_model_lineage_id;
    std::unordered_map<std::string, double> reward_component_sums;
};

class MetricEventJournal {
public:
    static constexpr std::size_t kDefaultEventCapacity = 4096;
    static constexpr std::size_t kDefaultByteCapacity = 16 * 1024 * 1024;

    MetricEventJournal(common::ContractIdentity contract,
                       common::SchemaIdentity schema,
                       common::ServiceInstanceIdentity source,
                       std::size_t capacity = kDefaultEventCapacity,
                       std::size_t byte_capacity = kDefaultByteCapacity,
                       std::chrono::milliseconds flush_interval =
                           std::chrono::milliseconds(5000));

    const common::ServiceInstanceIdentity& source() const { return source_; }

    bool AppendEpisode(training::EpisodeMetricFact fact,
                       int64_t committed_at_unix_ms);
    void Finalize(int64_t finalized_at_unix_ms);
    bool WaitForFinalAcknowledgement(
        std::chrono::milliseconds timeout);
    void Get(const training::GetMetricBatchReq& request,
             training::GetMetricBatchRsp& response);
    void Ack(const training::AckMetricBatchReq& request,
             training::AckMetricBatchRsp& response);

private:
    bool ValidContract(const common::ContractIdentity& contract) const;
    bool ValidConsumer(
        const common::ServiceInstanceIdentity& consumer) const;
    bool SameConsumer(
        const common::ServiceInstanceIdentity& consumer) const;
    bool CursorMatchesCommitted(
        const training::MetricBatchCursor& cursor) const;
    bool ReadyToSeal(
        std::chrono::steady_clock::time_point now) const;
    std::chrono::steady_clock::time_point NextSealDeadline() const;
    void FillAvailability(training::GetMetricBatchRsp& response) const;
    void FillAvailability(training::AckMetricBatchRsp& response) const;
    bool BuildPendingBatch(const training::GetMetricBatchReq& request,
                           int64_t now_unix_ms,
                           std::string& error);
    static std::string BatchDigest(const training::MetricBatch& batch);

    common::ContractIdentity contract_;
    common::SchemaIdentity schema_;
    common::ServiceInstanceIdentity source_;
    std::size_t capacity_;
    std::size_t byte_capacity_;
    std::chrono::milliseconds flush_interval_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<training::MetricEvent> events_;
    std::deque<std::chrono::steady_clock::time_point> event_enqueued_at_;
    std::size_t event_bytes_ = 0;
    uint64_t next_event_sequence_ = 1;
    uint64_t next_batch_sequence_ = 1;
    training::MetricBatchCursor committed_cursor_;
    std::optional<training::MetricBatch> pending_batch_;
    std::optional<common::ServiceInstanceIdentity> consumer_;
    bool source_final_ = false;
    bool final_batch_acknowledged_ = false;
    uint64_t final_event_sequence_ = 0;
    int64_t final_watermark_unix_ms_ = 0;
    int64_t last_event_committed_at_unix_ms_ = 0;
    int64_t last_acked_watermark_unix_ms_ = 0;
    std::chrono::steady_clock::time_point last_batch_created_at_;
};

class EpisodeMetricsWindow {
public:
    explicit EpisodeMetricsWindow(std::size_t capacity);

    void AddCompleted(maze::EpisodeMode episode_mode,
                      std::vector<AgentEpisodeResult> agents);
    void AddExcluded(maze::EpisodeMode episode_mode,
                     std::size_t agent_count,
                     maze::MazeTerminationReason reason);
    void Fill(training::MetricSnapshot* snapshot,
              const common::ServiceInstanceIdentity& source,
              uint64_t sequence,
              int64_t timestamp_unix_ms) const;

private:
    struct Entry {
        bool excluded = false;
        maze::EpisodeMode episode_mode = maze::EPISODE_MODE_UNSPECIFIED;
        std::vector<AgentEpisodeResult> agents;
    };

    void Push(Entry entry);
    void PushTraining(Entry entry);

    std::size_t capacity_;
    mutable std::mutex mutex_;
    std::deque<Entry> entries_;
    std::deque<Entry> training_entries_;
    uint64_t completed_training_episode_count_ = 0;
};
