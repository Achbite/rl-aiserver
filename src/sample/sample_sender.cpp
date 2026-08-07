#include "sample/sample_sender.h"

#include "log/logger.h"

#include <algorithm>
#include <chrono>

namespace {

double ElapsedMs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

}  // namespace

SampleSender::SampleSender(const AIServerConfig& config)
    : config_(config.sample_output),
      contract_(config.contract),
      producer_fragment_reserve_(
          static_cast<std::size_t>(std::max(1, config.task.agent_num))) {}

SampleSender::~SampleSender() {
    StopAndDrain();
}

bool SampleSender::ProbeDistributor() {
    training::DistributorStatusReq request;
    training::DistributorStatusRsp response;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::milliseconds(config_.health_timeout_ms));
    grpc::Status status = stub_->GetStatus(&context, request, &response);
    if (!status.ok()) {
        MarkDegraded(
            "sample ingress status failed: " + status.error_message());
        return false;
    }
    const auto& contract = response.contract();
    const bool contract_matches =
        contract.package_name() == contract_.package_name &&
        contract.package_version() == contract_.package_version &&
        contract.source_digest().algorithm() ==
            common::DIGEST_ALGORITHM_SHA256 &&
        contract.source_digest().hex() == contract_.source_digest.hex &&
        contract.artifact_digest().algorithm() ==
            common::DIGEST_ALGORITHM_SHA256 &&
        contract.artifact_digest().hex() == contract_.artifact_digest.hex &&
        contract.platform() == contract_.platform &&
        contract.generator_identity() == contract_.generator_identity;
    if (!contract_matches ||
        response.distributor().component() != "sample-distributor" ||
        response.distributor().instance_id().empty() ||
        response.distributor().lifecycle_epoch() == 0) {
        MarkDegraded("sample ingress identity does not match rl-contracts 0.10.0");
        return false;
    }
    if (!response.ready() || !response.ingress_ready()) {
        MarkDegraded("sample ingress is not ready");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    distributor_instance_id_ = response.distributor().instance_id();
    ready_ = true;
    degraded_ = false;
    last_error_.clear();
    return true;
}

bool SampleSender::Start() {
    if (!config_.enabled) {
        MarkDegraded("sample output is disabled");
        return false;
    }

    std::string target = config_.host + ":" + std::to_string(config_.port);
    channel_ = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
    stub_ = training::SampleDistributorService::NewStub(channel_);
    auto deadline = std::chrono::system_clock::now() +
                    std::chrono::milliseconds(config_.health_timeout_ms);
    if (!channel_->WaitForConnected(deadline)) {
        MarkDegraded("sample ingress connection timeout: " + target);
        return false;
    }
    if (!ProbeDistributor()) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        accepting_ = true;
        stop_requested_ = false;
        force_stop_ = false;
        training_capacity_wait_ = false;
        training_capacity_retry_after_ms_ = 0;
    }
    sender_thread_ = std::thread(&SampleSender::SenderLoop, this);
    LOG_INFO("SampleSender", "就绪: target=%s, distributor_instance_id=%s",
             target.c_str(), distributor_instance_id_.c_str());
    return true;
}

bool SampleSender::Enqueue(const training::SampleBatch& batch) {
    QueueItem item;
    item.batch = batch;
    item.samples = batch.samples_size();
    item.estimated_bytes = static_cast<int64_t>(batch.ByteSizeLong());

    std::unique_lock<std::mutex> lock(mutex_);
    auto has_capacity = [this, &item]() {
        return force_stop_ || !accepting_ || degraded_ ||
               (queue_.size() < config_.outbound_max_fragments &&
                queue_estimated_bytes_ + item.estimated_bytes <=
                    static_cast<int64_t>(config_.outbound_max_estimated_bytes));
    };
    if (!space_cv_.wait_for(lock,
                            std::chrono::milliseconds(config_.enqueue_timeout_ms),
                            has_capacity)) {
        degraded_ = true;
        last_error_ = "outbound queue enqueue timeout";
        return false;
    }
    if (!accepting_ || force_stop_ || degraded_) {
        return false;
    }

    queue_.push_back(std::move(item));
    queue_samples_ += queue_.back().samples;
    queue_estimated_bytes_ += queue_.back().estimated_bytes;
    queue_high_watermark_ =
        std::max(queue_high_watermark_, static_cast<int64_t>(queue_.size()));
    queue_cv_.notify_one();
    return true;
}

SampleSender::SendResult SampleSender::SendFront(
    const QueueItem& item,
    bool& duplicate,
    int& attempts_used,
    int& retry_after_ms,
    std::string& error) {
    duplicate = false;
    attempts_used = 0;
    retry_after_ms = 0;
    const int attempts = std::max(1, config_.max_attempts);

    for (int attempt = 0; attempt < attempts; ++attempt) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (force_stop_) {
                error = "sender forced to stop";
                return SendResult::kRejected;
            }
            ++credit_request_count_;
            ++attempts_used;
            if (item.attempts + attempt > 0) ++retry_attempt_count_;
        }

        training::AcquireSampleCreditReq credit_request;
        credit_request.set_request_id(item.batch.batch_id() + "-credit");
        *credit_request.mutable_producer() = item.batch.producer();
        *credit_request.mutable_contract() = item.batch.contract();
        credit_request.set_batch_id(item.batch.batch_id());
        *credit_request.mutable_payload_digest() = item.batch.payload_digest();
        *credit_request.mutable_behavior_policy() =
            item.batch.behavior_policy();
        *credit_request.mutable_training_semantics() =
            item.batch.training_semantics();
        credit_request.set_sample_count(item.samples);
        credit_request.set_fragment_count(1);
        credit_request.set_estimated_bytes(item.batch.ByteSizeLong());
        credit_request.set_created_at_unix_ms(
            item.batch.created_at_unix_ms());

        training::SampleCreditGrant credit;
        grpc::ClientContext credit_context;
        credit_context.set_deadline(
            std::chrono::system_clock::now() +
            std::chrono::milliseconds(config_.rpc_timeout_ms));
        {
            std::lock_guard<std::mutex> rpc_lock(rpc_mutex_);
            active_rpc_ = &credit_context;
        }
        grpc::Status credit_status =
            stub_->AcquireSampleCredit(&credit_context, credit_request, &credit);
        {
            std::lock_guard<std::mutex> rpc_lock(rpc_mutex_);
            active_rpc_ = nullptr;
        }
        if (!credit_status.ok()) {
            error = credit_status.error_message();
            if (attempt + 1 < attempts) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(50 * (attempt + 1)));
                continue;
            }
            return SendResult::kRejected;
        }
        if (credit.result() != training::SAMPLE_CREDIT_RESULT_GRANTED) {
            switch (credit.result()) {
                case training::SAMPLE_CREDIT_RESULT_WAIT_NO_DEMAND:
                case training::SAMPLE_CREDIT_RESULT_WAIT_INFLIGHT_LIMIT:
                case training::SAMPLE_CREDIT_RESULT_WAIT_CAPACITY:
                case training::SAMPLE_CREDIT_RESULT_WAIT_DRAINING: {
                    std::lock_guard<std::mutex> lock(mutex_);
                    ++credit_wait_count_;
                    retry_after_ms = std::max(1, credit.retry_after_ms());
                    training_capacity_retry_after_ms_ = retry_after_ms;
                    if (!training_capacity_wait_) {
                        training_capacity_wait_ = true;
                        capacity_wait_started_ =
                            std::chrono::steady_clock::now();
                    }
                    last_error_.clear();
                    return SendResult::kWait;
                }
                case training::SAMPLE_CREDIT_RESULT_REJECTED_FRESHNESS: {
                    error = credit.message();
                    return SendResult::kProducerStale;
                }
                default:
                    error = credit.message();
                    return SendResult::kRejected;
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++credit_grant_count_;
            if (attempt > 0 || item.attempts > 0) ++credit_reacquire_count_;
            ++push_attempt_count_;
        }

        training::PushSamplesRsp response;
        training::PushSamplesReq push_request;
        push_request.set_credit_id(credit.credit_id());
        *push_request.mutable_batch() = item.batch;
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() +
                             std::chrono::milliseconds(config_.rpc_timeout_ms));
        {
            std::lock_guard<std::mutex> rpc_lock(rpc_mutex_);
            active_rpc_ = &context;
        }

        auto start = std::chrono::steady_clock::now();
        grpc::Status status =
            stub_->PushSamples(&context, push_request, &response);
        double latency_ms = ElapsedMs(start);
        {
            std::lock_guard<std::mutex> rpc_lock(rpc_mutex_);
            active_rpc_ = nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++push_rpc_count_;
            push_rpc_latency_sum_ms_ += latency_ms;
            push_rpc_latency_max_ms_ = std::max(push_rpc_latency_max_ms_, latency_ms);
        }

        if (status.ok() &&
            (response.result() == training::PUSH_RESULT_ACCEPTED ||
             response.result() == training::PUSH_RESULT_DUPLICATE)) {
            duplicate = response.result() == training::PUSH_RESULT_DUPLICATE;
            std::lock_guard<std::mutex> lock(mutex_);
            if (training_capacity_wait_) {
                capacity_wait_ms_ +=
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() -
                        capacity_wait_started_)
                        .count();
                training_capacity_wait_ = false;
                training_capacity_retry_after_ms_ = 0;
            }
            return SendResult::kCommitted;
        }

        if (status.ok()) {
            error = response.message();
            std::lock_guard<std::mutex> lock(mutex_);
            ++rejected_push_attempt_count_;
            if (response.result() == training::PUSH_RESULT_REJECTED_INVALID ||
                response.result() == training::PUSH_RESULT_REJECTED_IDENTITY) {
                if (response.result() ==
                        training::PUSH_RESULT_REJECTED_IDENTITY &&
                    attempt + 1 < attempts) {
                    continue;
                }
                return SendResult::kRejected;
            }
        } else {
            error = status.error_message();
        }

        if (attempt + 1 < attempts) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(50 * (attempt + 1)));
        }
    }
    return SendResult::kRejected;
}

void SampleSender::SenderLoop() {
    while (true) {
        QueueItem item;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            queue_cv_.wait(lock, [this]() {
                return force_stop_ || !queue_.empty() || stop_requested_;
            });
            if (force_stop_ || (stop_requested_ && queue_.empty())) {
                break;
            }
            item = queue_.front();
        }

        bool duplicate = false;
        int attempts_used = 0;
        int retry_after_ms = 0;
        std::string error;
        const SendResult result = SendFront(
            item, duplicate, attempts_used, retry_after_ms, error);
        if (result == SendResult::kCommitted) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!queue_.empty() &&
                queue_.front().batch.batch_id() == item.batch.batch_id()) {
                queue_samples_ -= queue_.front().samples;
                queue_estimated_bytes_ -= queue_.front().estimated_bytes;
                queue_.pop_front();
                if (duplicate) {
                    ++duplicate_push_attempt_count_;
                } else {
                    accepted_unique_samples_ += item.samples;
                    ++accepted_unique_batches_;
                }
                space_cv_.notify_all();
                if (queue_.empty()) drained_cv_.notify_all();
            }
            continue;
        }

        if (result == SendResult::kWait) {
            std::unique_lock<std::mutex> lock(mutex_);
            queue_cv_.wait_for(
                lock, std::chrono::milliseconds(std::max(1, retry_after_ms)),
                [this]() { return force_stop_; });
            continue;
        }

        if (result == SendResult::kProducerStale) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!queue_.empty() &&
                queue_.front().batch.batch_id() == item.batch.batch_id()) {
                queue_samples_ -= queue_.front().samples;
                queue_estimated_bytes_ -= queue_.front().estimated_bytes;
                producer_stale_count_ += queue_.front().samples;
                producer_stale_samples_by_model_[static_cast<int>(
                    queue_.front().batch.behavior_policy().model_version())] +=
                    queue_.front().samples;
                queue_.pop_front();
                if (training_capacity_wait_) {
                    capacity_wait_ms_ +=
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() -
                            capacity_wait_started_)
                            .count();
                    training_capacity_wait_ = false;
                    training_capacity_retry_after_ms_ = 0;
                }
                last_error_.clear();
                space_cv_.notify_all();
                if (queue_.empty()) drained_cv_.notify_all();
            }
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!queue_.empty() &&
                queue_.front().batch.batch_id() == item.batch.batch_id()) {
                queue_.front().attempts += attempts_used;
            }
        }
        MarkDegraded(error.empty() ? "PushSamples failed" : error);
        std::unique_lock<std::mutex> lock(mutex_);
        queue_cv_.wait_for(lock, std::chrono::seconds(1), [this]() {
            return force_stop_;
        });
    }

    std::lock_guard<std::mutex> lock(mutex_);
    ready_ = false;
    drained_cv_.notify_all();
}

void SampleSender::CancelActiveRpc() {
    std::lock_guard<std::mutex> lock(rpc_mutex_);
    if (active_rpc_) {
        active_rpc_->TryCancel();
    }
}

bool SampleSender::StopAndDrain() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!sender_thread_.joinable()) {
            accepting_ = false;
            ready_ = false;
            return queue_.empty();
        }
        accepting_ = false;
        stop_requested_ = true;
        queue_cv_.notify_all();
    }

    bool drained = false;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        drained = drained_cv_.wait_for(
            lock, std::chrono::milliseconds(config_.drain_timeout_ms),
            [this]() { return queue_.empty(); });
        if (!drained) {
            for (const auto& item : queue_) {
                final_drop_unique_samples_ += item.samples;
                ++final_drop_unique_batches_;
            }
            queue_.clear();
            queue_samples_ = 0;
            queue_estimated_bytes_ = 0;
            degraded_ = true;
            last_error_ = "outbound drain deadline exceeded";
            force_stop_ = true;
            queue_cv_.notify_all();
            space_cv_.notify_all();
        }
    }
    if (!drained) CancelActiveRpc();
    if (sender_thread_.joinable()) sender_thread_.join();
    return drained;
}

bool SampleSender::IsReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ready_;
}

bool SampleSender::IsDegraded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return degraded_;
}

bool SampleSender::IsWaitingForTrainingCapacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ProducerCapacityConstrainedLocked();
}

int SampleSender::TrainingCapacityRetryAfterMs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (training_capacity_retry_after_ms_ > 0) {
        return training_capacity_retry_after_ms_;
    }
    return std::max(1, config_.enqueue_timeout_ms);
}

bool SampleSender::ProducerCapacityConstrainedLocked() const {
    if (training_capacity_wait_) return true;

    // One Client Update can close one fragment for every assigned Agent. Keep
    // that whole frame atomic: once the reserve is unavailable, the RPC layer
    // asks the Client to retry the same frame instead of partially mutating it.
    if (queue_.size() + producer_fragment_reserve_ >
        config_.outbound_max_fragments) {
        return true;
    }

    const auto bytes_per_slot =
        config_.outbound_max_estimated_bytes /
        config_.outbound_max_fragments;
    const auto byte_reserve =
        bytes_per_slot * producer_fragment_reserve_;
    return byte_reserve > config_.outbound_max_estimated_bytes ||
           queue_estimated_bytes_ >
               static_cast<int64_t>(
                   config_.outbound_max_estimated_bytes - byte_reserve);
}

void SampleSender::MarkDegraded(const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (training_capacity_wait_) {
        capacity_wait_ms_ +=
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - capacity_wait_started_)
                .count();
        training_capacity_wait_ = false;
        training_capacity_retry_after_ms_ = 0;
    }
    degraded_ = true;
    last_error_ = error;
    space_cv_.notify_all();
}

void SampleSender::RecordFinalDrop(int64_t samples, int64_t batches,
                                   const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    final_drop_unique_samples_ += samples;
    final_drop_unique_batches_ += batches;
    degraded_ = true;
    last_error_ = error;
}

SampleSender::Snapshot SampleSender::GetSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Snapshot snapshot;
    snapshot.ready = ready_;
    snapshot.degraded = degraded_;
    snapshot.training_capacity_wait = ProducerCapacityConstrainedLocked();
    snapshot.retry_after_ms =
        training_capacity_retry_after_ms_ > 0
            ? training_capacity_retry_after_ms_
            : std::max(1, config_.enqueue_timeout_ms);
    snapshot.queue_fragments = queue_.size();
    snapshot.queue_samples = queue_samples_;
    snapshot.queue_estimated_bytes = queue_estimated_bytes_;
    snapshot.queue_high_watermark = queue_high_watermark_;
    snapshot.push_attempt_count = push_attempt_count_;
    snapshot.accepted_unique_samples = accepted_unique_samples_;
    snapshot.accepted_unique_batches = accepted_unique_batches_;
    snapshot.duplicate_push_attempt_count = duplicate_push_attempt_count_;
    snapshot.rejected_push_attempt_count = rejected_push_attempt_count_;
    snapshot.retry_attempt_count = retry_attempt_count_;
    snapshot.final_drop_unique_samples = final_drop_unique_samples_;
    snapshot.final_drop_unique_batches = final_drop_unique_batches_;
    snapshot.push_rpc_count = push_rpc_count_;
    snapshot.push_rpc_latency_sum_ms = push_rpc_latency_sum_ms_;
    snapshot.push_rpc_latency_max_ms = push_rpc_latency_max_ms_;
    snapshot.credit_request_count = credit_request_count_;
    snapshot.credit_grant_count = credit_grant_count_;
    snapshot.credit_wait_count = credit_wait_count_;
    snapshot.credit_reacquire_count = credit_reacquire_count_;
    snapshot.producer_stale_count = producer_stale_count_;
    snapshot.producer_stale_samples_by_model =
        producer_stale_samples_by_model_;
    snapshot.capacity_wait_ms = capacity_wait_ms_;
    if (training_capacity_wait_) {
        snapshot.capacity_wait_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - capacity_wait_started_)
                .count();
    }
    snapshot.distributor_instance_id = distributor_instance_id_;
    snapshot.last_error = last_error_;
    return snapshot;
}
