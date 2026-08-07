#pragma once

#include "config/config_loader.h"
#include "contracts/contract_namespaces.h"
#include "training.grpc.pb.h"

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

class SampleSender {
public:
    struct Snapshot {
        bool ready = false;
        bool degraded = false;
        bool training_capacity_wait = false;
        int retry_after_ms = 0;
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
        int64_t push_rpc_count = 0;
        double push_rpc_latency_sum_ms = 0.0;
        double push_rpc_latency_max_ms = 0.0;
        int64_t credit_request_count = 0;
        int64_t credit_grant_count = 0;
        int64_t credit_wait_count = 0;
        int64_t credit_reacquire_count = 0;
        int64_t producer_stale_count = 0;
        std::unordered_map<int, int64_t> producer_stale_samples_by_model;
        int64_t capacity_wait_ms = 0;
        std::string distributor_instance_id;
        std::string last_error;
    };

    explicit SampleSender(const AIServerConfig& config);
    ~SampleSender();

    bool Start();
    bool Enqueue(const training::SampleBatch& batch);
    bool StopAndDrain();

    bool IsReady() const;
    bool IsDegraded() const;
    bool IsWaitingForTrainingCapacity() const;
    int TrainingCapacityRetryAfterMs() const;
    void MarkDegraded(const std::string& error);
    void RecordFinalDrop(int64_t samples, int64_t batches,
                         const std::string& error);
    Snapshot GetSnapshot() const;

private:
    enum class SendResult {
        kCommitted,
        kWait,
        kProducerStale,
        kRejected,
    };

    struct QueueItem {
        training::SampleBatch batch;
        int64_t samples = 0;
        int64_t estimated_bytes = 0;
        int attempts = 0;
    };

    bool ProbeDistributor();
    void SenderLoop();
    SendResult SendFront(const QueueItem& item, bool& duplicate,
                         int& attempts_used, int& retry_after_ms,
                         std::string& error);
    void CancelActiveRpc();

    SampleOutputConfig config_;
    ContractConfig contract_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<training::SampleDistributorService::Stub> stub_;

    mutable std::mutex mutex_;
    std::condition_variable queue_cv_;
    std::condition_variable space_cv_;
    std::condition_variable drained_cv_;
    std::deque<QueueItem> queue_;
    int64_t queue_samples_ = 0;
    int64_t queue_estimated_bytes_ = 0;
    bool accepting_ = false;
    bool stop_requested_ = false;
    bool force_stop_ = false;
    bool ready_ = false;
    bool degraded_ = false;
    bool training_capacity_wait_ = false;
    int training_capacity_retry_after_ms_ = 0;
    std::chrono::steady_clock::time_point capacity_wait_started_{};
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
    int64_t push_rpc_count_ = 0;
    double push_rpc_latency_sum_ms_ = 0.0;
    double push_rpc_latency_max_ms_ = 0.0;
    int64_t credit_request_count_ = 0;
    int64_t credit_grant_count_ = 0;
    int64_t credit_wait_count_ = 0;
    int64_t credit_reacquire_count_ = 0;
    int64_t producer_stale_count_ = 0;
    std::unordered_map<int, int64_t> producer_stale_samples_by_model_;
    int64_t capacity_wait_ms_ = 0;
    std::string distributor_instance_id_;
    std::string last_error_;
};
