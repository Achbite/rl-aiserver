#pragma once

#include "config/config_loader.h"
#include "contracts/contract_namespaces.h"
#include "training.grpc.pb.h"

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// AIServer-owned asynchronous producer-side distributor. It owns only the
// bounded local outbound queue, exact immutable retries, readiness and drain;
// SamplePool capacity, eviction and Learner batch assembly are remote facts.
class SampleDistributor {
public:
    enum class DeliveryState {
        kStarting,
        kHealthy,
        kLocalBackpressure,
        kTransientRetry,
        kTerminalFault,
        kDraining,
        kStopped,
    };

    enum class ReservationResult {
        kReserved,
        kRetryableUnavailable,
        kTerminalFault,
    };

    enum class CommitResult {
        kCommitted,
        kRetryableUnavailable,
        kTerminalFault,
    };

    enum class SealResult {
        kSealed,
        kRetryableUnavailable,
        kTerminalFault,
    };

    struct Snapshot {
        DeliveryState delivery_state = DeliveryState::kStopped;
        bool ready = false;
        bool degraded = false;
        bool transient_retry = false;
        bool backend_recovering = false;
        bool terminal_fault = false;
        bool local_backpressure = false;
        bool sample_delivery_paused = false;
        int retry_after_ms = 0;
        int64_t recovery_elapsed_ms = 0;
        std::size_t queue_fragments = 0;
        int64_t queue_samples = 0;
        int64_t queue_estimated_bytes = 0;
        int64_t queue_high_watermark = 0;
        int64_t push_attempt_count = 0;
        int64_t accepted_unique_samples = 0;
        int64_t accepted_unique_batches = 0;
        int64_t duplicate_push_attempt_count = 0;
        int64_t rejected_push_attempt_count = 0;
        int64_t retry_attempt_count = 0;
        int64_t final_drop_unique_samples = 0;
        int64_t final_drop_unique_batches = 0;
        int64_t unresolved_push_outcome_unknown_samples = 0;
        int64_t unresolved_push_outcome_unknown_batches = 0;
        int64_t push_rpc_count = 0;
        double push_rpc_latency_sum_ms = 0.0;
        double push_rpc_latency_max_ms = 0.0;
        std::string sample_pool_instance_id;
        std::string last_error;
    };

    explicit SampleDistributor(const AIServerConfig& config);
    ~SampleDistributor();

    bool Start();
    bool Enqueue(const training::SampleBatch& batch);
    ReservationResult ReserveEnqueueBatchSet(
        const std::vector<training::SampleBatch>& batches,
        uint64_t& reservation_id,
        std::string& error);
    SealResult SealEnqueueBatchSet(uint64_t reservation_id,
                                   std::string& error);
    CommitResult CommitEnqueueBatchSet(uint64_t reservation_id,
                                       std::string& error);
    void CancelEnqueueBatchSet(uint64_t reservation_id);
    bool HasEnqueueReservation(uint64_t reservation_id) const;
    bool StopAndDrain();

    bool IsReady() const;
    bool IsDegraded() const;
    bool IsTrainingDeliveryReady() const;
    bool IsPausedAtSafeBoundary() const;
    int PauseRetryAfterMs() const;
    void MarkDegraded(const std::string& error);
    void RecordFinalDrop(int64_t samples,
                         int64_t batches,
                         const std::string& error);
    Snapshot GetSnapshot() const;

private:
    friend struct MazeServiceUpdateTestAccess;

    enum class SendResult {
        kCommitted,
        kTransient,
        kRejected,
    };

    enum class StatusRefreshResult {
        kHealthy,
        kTransient,
        kTerminal,
    };

    struct QueueItem {
        training::SampleBatch batch;
        int64_t samples = 0;
        int64_t estimated_bytes = 0;
        int attempts = 0;
        bool push_outcome_unknown = false;
    };

    bool ProbeSamplePool();
    StatusRefreshResult RefreshSamplePoolStatus();
    bool ValidateSamplePoolStatus(
        const training::SamplePoolStatusRsp& response,
        std::string& error) const;
    void SenderLoop();
    SendResult SendFront(const QueueItem& item,
                         bool& duplicate,
                         int& attempts_used,
                         int& retry_after_ms,
                         std::string& error);
    void CancelActiveRpc();
    bool ProducerCapacityConstrainedLocked() const;
    bool MarkTransient(const std::string& error, int retry_after_ms);
    void MarkHealthy();
    void SetDeliveryStateLocked(DeliveryState state);

    SampleDistributorConfig config_;
    ContractConfig contract_;
    std::size_t producer_fragment_reserve_ = 1;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<training::SamplePoolIngressService::Stub> stub_;

    mutable std::mutex mutex_;
    std::condition_variable queue_cv_;
    std::condition_variable space_cv_;
    std::condition_variable drained_cv_;
    std::list<QueueItem> queue_;
    std::list<QueueItem> reserved_items_;
    uint64_t active_reservation_id_ = 0;
    uint64_t active_reservation_delivery_generation_ = 0;
    bool active_reservation_sealed_ = false;
    uint64_t next_reservation_id_ = 1;
    int64_t reserved_samples_ = 0;
    int64_t reserved_estimated_bytes_ = 0;
    int64_t queue_samples_ = 0;
    int64_t queue_estimated_bytes_ = 0;
    bool accepting_ = false;
    bool stop_requested_ = false;
    bool force_stop_ = false;
    bool ready_ = false;
    bool degraded_ = false;
    DeliveryState delivery_state_ = DeliveryState::kStopped;
    uint64_t delivery_generation_ = 0;
    int transient_retry_after_ms_ = 0;
    int status_failure_attempts_ = 0;
    bool recovery_active_ = false;
    std::chrono::steady_clock::time_point recovery_started_{};
    std::thread sender_thread_;

    mutable std::mutex rpc_mutex_;
    grpc::ClientContext* active_rpc_ = nullptr;

    int64_t queue_high_watermark_ = 0;
    int64_t push_attempt_count_ = 0;
    int64_t accepted_unique_samples_ = 0;
    int64_t accepted_unique_batches_ = 0;
    int64_t duplicate_push_attempt_count_ = 0;
    int64_t rejected_push_attempt_count_ = 0;
    int64_t retry_attempt_count_ = 0;
    int64_t final_drop_unique_samples_ = 0;
    int64_t final_drop_unique_batches_ = 0;
    int64_t unresolved_push_outcome_unknown_samples_ = 0;
    int64_t unresolved_push_outcome_unknown_batches_ = 0;
    int64_t push_rpc_count_ = 0;
    double push_rpc_latency_sum_ms_ = 0.0;
    double push_rpc_latency_max_ms_ = 0.0;
    std::string sample_pool_instance_id_;
    uint64_t sample_pool_lifecycle_epoch_ = 0;
    std::string last_error_;
};
