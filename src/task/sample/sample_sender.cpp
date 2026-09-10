#include "task/sample/sample_sender.h"

#include "log/logger.h"

#include <algorithm>
#include <chrono>
#include <limits>

namespace {

double ElapsedMs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

bool SameSamplePool(const common::ServiceInstanceIdentity& actual,
                    const std::string& expected_instance_id,
                    uint64_t expected_lifecycle_epoch) {
    return actual.instance_id() == expected_instance_id &&
           actual.lifecycle_epoch() == expected_lifecycle_epoch;
}

bool IsTerminalPushRejection(training::PushResult result) {
    return result == training::PUSH_RESULT_REJECTED_INVALID ||
           result == training::PUSH_RESULT_REJECTED_CONFLICT ||
           result == training::PUSH_RESULT_REJECTED_FINALIZED;
}

bool IsRetryableOutcomeUnknownTransport(const grpc::Status& status) {
    switch (status.error_code()) {
        case grpc::StatusCode::CANCELLED:
        case grpc::StatusCode::UNKNOWN:
        case grpc::StatusCode::DEADLINE_EXCEEDED:
        case grpc::StatusCode::RESOURCE_EXHAUSTED:
        case grpc::StatusCode::ABORTED:
        case grpc::StatusCode::INTERNAL:
        case grpc::StatusCode::UNAVAILABLE:
            return true;
        default:
            return false;
    }
}

int RetryBackoffMs(int attempts) {
    const int exponent = std::min(5, std::max(0, attempts - 1));
    return std::min(1000, 50 * (1 << exponent));
}

bool ValidatePushResponse(
    const training::ProcessedTransitionEnvelope& envelope,
    int64_t transition_count,
    const training::PushSamplesRsp& response,
    const std::string& sample_pool_instance_id,
    uint64_t sample_pool_lifecycle_epoch,
    std::string& error) {
    if (!SameSamplePool(response.sample_pool(), sample_pool_instance_id,
                        sample_pool_lifecycle_epoch)) {
        error = "PushSamples response SamplePool identity changed";
        return false;
    }
    if (response.envelope_id() != envelope.envelope_id()) {
        error = "PushSamples response does not echo the exact envelope_id";
        return false;
    }
    if (response.result() == training::PUSH_RESULT_ACCEPTED) {
        if (transition_count <= 0 ||
            transition_count != envelope.samples_size()) {
            error = "PushSamples ACCEPTED response has an invalid count";
            return false;
        }
        return true;
    }
    if (response.result() == training::PUSH_RESULT_DUPLICATE) {
        if (transition_count <= 0 ||
            transition_count != envelope.samples_size()) {
            error = "PushSamples DUPLICATE response has an invalid count";
            return false;
        }
        return true;
    }
    if (response.result() == training::PUSH_RESULT_REJECTED_CAPACITY ||
        IsTerminalPushRejection(response.result())) {
        return true;
    }
    error = "PushSamples response result is unspecified";
    return false;
}

}  // namespace

SampleDistributor::SampleDistributor(const SampleDistributorConfig& config)
    : config_(config) {}

SampleDistributor::~SampleDistributor() {
    StopAndDrain();
}

bool SampleDistributor::ProbeSamplePool() {
    training::SamplePoolStatusReq request;
    training::SamplePoolStatusRsp response;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::milliseconds(config_.health_timeout_ms));
    const grpc::Status status = stub_->GetStatus(&context, request, &response);
    if (!status.ok()) {
        MarkDegraded("SamplePool ingress status failed: " +
                     status.error_message());
        return false;
    }
    std::string error;
    if (!ValidateSamplePoolStatus(response, error)) {
        MarkDegraded(error);
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    sample_pool_instance_id_ = response.sample_pool().instance_id();
    sample_pool_lifecycle_epoch_ = response.sample_pool().lifecycle_epoch();
    ready_ = true;
    degraded_ = false;
    SetDeliveryStateLocked(DeliveryState::kHealthy);
    transient_retry_after_ms_ = 0;
    last_error_.clear();
    return true;
}

bool SampleDistributor::ValidateSamplePoolStatus(
    const training::SamplePoolStatusRsp& response,
    std::string& error) const {
    if (response.sample_pool().component().empty() ||
        response.sample_pool().instance_id().empty() ||
        response.sample_pool().lifecycle_epoch() == 0) {
        error = "SamplePool ingress lifecycle identity is invalid";
        return false;
    }
    // pool_ready is a data-availability fact. An empty Pool remains a valid
    // ingress, so producer readiness depends only on the service and ingress.
    if (!response.ready() || !response.ingress_ready() ||
        response.finalized()) {
        error = "SamplePool ingress is not accepting transitions";
        return false;
    }
    return true;
}

SampleDistributor::StatusRefreshResult
SampleDistributor::RefreshSamplePoolStatus() {
    training::SamplePoolStatusReq request;
    training::SamplePoolStatusRsp response;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::milliseconds(config_.rpc_timeout_ms));
    {
        std::lock_guard<std::mutex> rpc_lock(rpc_mutex_);
        active_rpc_ = &context;
    }
    const grpc::Status status = stub_->GetStatus(&context, request, &response);
    {
        std::lock_guard<std::mutex> rpc_lock(rpc_mutex_);
        active_rpc_ = nullptr;
    }
    if (!status.ok()) {
        const std::string error = "SamplePool ingress status refresh failed: " +
                                  status.error_message();
        if (IsRetryableOutcomeUnknownTransport(status)) {
            int attempts = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++status_failure_attempts_;
                attempts = status_failure_attempts_;
            }
            return MarkTransient(error, RetryBackoffMs(attempts))
                       ? StatusRefreshResult::kTransient
                       : StatusRefreshResult::kTerminal;
        }
        MarkDegraded(error);
        return StatusRefreshResult::kTerminal;
    }

    std::string error;
    if (!ValidateSamplePoolStatus(response, error)) {
        MarkDegraded(error);
        return StatusRefreshResult::kTerminal;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!SameSamplePool(response.sample_pool(), sample_pool_instance_id_,
                            sample_pool_lifecycle_epoch_)) {
            error = "SamplePool identity changed during training";
        }
    }
    if (!error.empty()) {
        MarkDegraded(error);
        return StatusRefreshResult::kTerminal;
    }
    MarkHealthy();
    return StatusRefreshResult::kHealthy;
}

bool SampleDistributor::Start() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        SetDeliveryStateLocked(DeliveryState::kStarting);
    }
    if (!config_.enabled) {
        MarkDegraded("AIServer SampleDistributor is disabled");
        return false;
    }

    const std::string target =
        config_.host + ":" + std::to_string(config_.port);
    channel_ = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
    stub_ = training::SamplePoolIngressService::NewStub(channel_);
    const auto deadline = std::chrono::system_clock::now() +
                          std::chrono::milliseconds(config_.health_timeout_ms);
    if (!channel_->WaitForConnected(deadline)) {
        MarkDegraded("SamplePool ingress connection timeout: " + target);
        return false;
    }
    if (!ProbeSamplePool()) return false;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        accepting_ = true;
        stop_requested_ = false;
        force_stop_ = false;
        recovery_active_ = false;
        transient_retry_after_ms_ = 0;
        status_failure_attempts_ = 0;
        SetDeliveryStateLocked(DeliveryState::kHealthy);
    }
    sender_thread_ = std::thread(&SampleDistributor::SenderLoop, this);
    LOG_INFO("SampleDistributor",
             "就绪: ingress=%s sample_pool_instance_id=%s",
             target.c_str(), sample_pool_instance_id_.c_str());
    return true;
}

bool SampleDistributor::Enqueue(
    const training::ProcessedTransitionEnvelope& envelope) {
    QueueItem item;
    item.envelope = envelope;
    item.transitions = envelope.samples_size();
    item.estimated_bytes = static_cast<int64_t>(envelope.ByteSizeLong());
    if (item.transitions <= 0 || item.estimated_bytes <= 0) return false;

    std::unique_lock<std::mutex> lock(mutex_);
    const auto has_capacity = [this, &item]() {
        return force_stop_ || !accepting_ || degraded_ ||
               (queue_.size() + reserved_items_.size() <
                    config_.outbound_max_envelopes &&
                queue_estimated_bytes_ + reserved_estimated_bytes_ +
                        item.estimated_bytes <=
                    static_cast<int64_t>(
                        config_.outbound_max_estimated_bytes));
    };
    if (!space_cv_.wait_for(
            lock, std::chrono::milliseconds(config_.enqueue_timeout_ms),
            has_capacity)) {
        return false;
    }
    if (!accepting_ || force_stop_ || degraded_) return false;

    queue_.push_back(std::move(item));
    queue_transitions_ += queue_.back().transitions;
    queue_estimated_bytes_ += queue_.back().estimated_bytes;
    queue_high_watermark_ =
        std::max(queue_high_watermark_, static_cast<int64_t>(queue_.size()));
    queue_cv_.notify_one();
    return true;
}

SampleDistributor::ReservationResult
SampleDistributor::ReserveEnqueueEnvelopeSet(
    const std::vector<training::ProcessedTransitionEnvelope>& envelopes,
    uint64_t& reservation_id,
    std::string& error) {
    reservation_id = 0;
    error.clear();
    if (envelopes.empty()) return ReservationResult::kReserved;

    std::list<QueueItem> items;
    int64_t transitions = 0;
    int64_t estimated_bytes = 0;
    for (const auto& envelope : envelopes) {
        QueueItem item;
        item.envelope = envelope;
        item.transitions = envelope.samples_size();
        item.estimated_bytes =
            static_cast<int64_t>(envelope.ByteSizeLong());
        if (item.transitions <= 0 || item.estimated_bytes <= 0 ||
            transitions >
                std::numeric_limits<int64_t>::max() - item.transitions ||
            estimated_bytes >
                std::numeric_limits<int64_t>::max() - item.estimated_bytes) {
            error = "processed envelope reservation is invalid";
            return ReservationResult::kTerminalFault;
        }
        transitions += item.transitions;
        estimated_bytes += item.estimated_bytes;
        items.push_back(std::move(item));
    }

    std::unique_lock<std::mutex> lock(mutex_);
    const auto has_capacity = [this, &items, estimated_bytes]() {
        return force_stop_ || !accepting_ || degraded_ ||
               delivery_state_ != DeliveryState::kHealthy ||
               (reserved_items_.empty() &&
                queue_.size() + items.size() <=
                    config_.outbound_max_envelopes &&
                queue_estimated_bytes_ + estimated_bytes <=
                    static_cast<int64_t>(
                        config_.outbound_max_estimated_bytes));
    };
    if (!space_cv_.wait_for(
            lock, std::chrono::milliseconds(config_.enqueue_timeout_ms),
            has_capacity)) {
        error = "outbound envelope-set reservation timeout";
        return ReservationResult::kRetryableUnavailable;
    }
    if (!accepting_ || force_stop_ || degraded_ || !ready_ ||
        delivery_state_ == DeliveryState::kTerminalFault ||
        delivery_state_ == DeliveryState::kDraining ||
        delivery_state_ == DeliveryState::kStopped ||
        delivery_state_ == DeliveryState::kStarting) {
        error = last_error_.empty()
                    ? "outbound queue is not accepting reservations"
                    : last_error_;
        return ReservationResult::kTerminalFault;
    }
    if (delivery_state_ != DeliveryState::kHealthy) {
        error = delivery_state_ == DeliveryState::kTransientRetry
                    ? "sample delivery is recovering"
                    : "sample delivery is locally backpressured";
        return ReservationResult::kRetryableUnavailable;
    }
    if (!reserved_items_.empty()) {
        error = "another outbound envelope-set reservation is active";
        return ReservationResult::kTerminalFault;
    }

    active_reservation_id_ = next_reservation_id_++;
    if (active_reservation_id_ == 0) {
        active_reservation_id_ = next_reservation_id_++;
    }
    reserved_items_.splice(reserved_items_.end(), items);
    reserved_transitions_ = transitions;
    reserved_estimated_bytes_ = estimated_bytes;
    active_reservation_delivery_generation_ = delivery_generation_;
    active_reservation_sealed_ = false;
    reservation_id = active_reservation_id_;
    return ReservationResult::kReserved;
}

SampleDistributor::SealResult SampleDistributor::SealEnqueueEnvelopeSet(
    uint64_t reservation_id,
    std::string& error) {
    error.clear();
    if (reservation_id == 0) return SealResult::kSealed;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto release_reservation = [this]() {
        reserved_items_.clear();
        reserved_transitions_ = 0;
        reserved_estimated_bytes_ = 0;
        active_reservation_id_ = 0;
        active_reservation_delivery_generation_ = 0;
        active_reservation_sealed_ = false;
        space_cv_.notify_all();
    };
    if (reservation_id != active_reservation_id_ ||
        reserved_items_.empty()) {
        degraded_ = true;
        SetDeliveryStateLocked(DeliveryState::kTerminalFault);
        error = "outbound reservation identity invariant failed before seal";
        last_error_ = error;
        release_reservation();
        return SealResult::kTerminalFault;
    }
    if (degraded_ || force_stop_ || !accepting_ || !ready_ ||
        delivery_state_ == DeliveryState::kTerminalFault ||
        delivery_state_ == DeliveryState::kDraining ||
        delivery_state_ == DeliveryState::kStopped ||
        delivery_state_ == DeliveryState::kStarting) {
        error = last_error_.empty()
                    ? "sample delivery became terminal before seal"
                    : last_error_;
        release_reservation();
        return SealResult::kTerminalFault;
    }
    if (delivery_state_ != DeliveryState::kHealthy ||
        active_reservation_delivery_generation_ != delivery_generation_) {
        error = delivery_state_ == DeliveryState::kTransientRetry
                    ? "sample delivery became transient before seal"
                    : "sample delivery state changed before seal";
        release_reservation();
        return SealResult::kRetryableUnavailable;
    }
    active_reservation_sealed_ = true;
    return SealResult::kSealed;
}

SampleDistributor::CommitResult SampleDistributor::CommitEnqueueEnvelopeSet(
    uint64_t reservation_id,
    std::string& error) {
    error.clear();
    if (reservation_id == 0) return CommitResult::kCommitted;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto release_reservation = [this]() {
        reserved_items_.clear();
        reserved_transitions_ = 0;
        reserved_estimated_bytes_ = 0;
        active_reservation_id_ = 0;
        active_reservation_delivery_generation_ = 0;
        active_reservation_sealed_ = false;
        space_cv_.notify_all();
    };
    if (active_reservation_sealed_ &&
        reservation_id != active_reservation_id_) {
        degraded_ = true;
        SetDeliveryStateLocked(DeliveryState::kTerminalFault);
        error = "sealed outbound reservation identity invariant failed";
        last_error_ = error;
        release_reservation();
        return CommitResult::kTerminalFault;
    }
    if (active_reservation_sealed_) {
        queue_.splice(queue_.end(), reserved_items_);
        queue_transitions_ += reserved_transitions_;
        queue_estimated_bytes_ += reserved_estimated_bytes_;
        queue_high_watermark_ = std::max(
            queue_high_watermark_, static_cast<int64_t>(queue_.size()));
        release_reservation();
        queue_cv_.notify_one();
        return CommitResult::kCommitted;
    }
    if (reservation_id != active_reservation_id_) {
        degraded_ = true;
        SetDeliveryStateLocked(DeliveryState::kTerminalFault);
        error = "outbound reservation identity invariant failed";
        last_error_ = error;
        release_reservation();
        return CommitResult::kTerminalFault;
    }
    if (degraded_ || force_stop_ || !accepting_ || !ready_ ||
        delivery_state_ == DeliveryState::kTerminalFault ||
        delivery_state_ == DeliveryState::kDraining ||
        delivery_state_ == DeliveryState::kStopped ||
        delivery_state_ == DeliveryState::kStarting) {
        error = last_error_.empty()
                    ? "sample delivery became terminal before commit"
                    : last_error_;
        release_reservation();
        return CommitResult::kTerminalFault;
    }
    if (delivery_state_ != DeliveryState::kHealthy ||
        active_reservation_delivery_generation_ != delivery_generation_) {
        error = delivery_state_ == DeliveryState::kTransientRetry
                    ? "sample delivery became transient before commit"
                    : "sample delivery state changed before commit";
        release_reservation();
        return CommitResult::kRetryableUnavailable;
    }
    queue_.splice(queue_.end(), reserved_items_);
    queue_transitions_ += reserved_transitions_;
    queue_estimated_bytes_ += reserved_estimated_bytes_;
    queue_high_watermark_ =
        std::max(queue_high_watermark_, static_cast<int64_t>(queue_.size()));
    release_reservation();
    queue_cv_.notify_one();
    return CommitResult::kCommitted;
}

bool SampleDistributor::HasEnqueueReservation(
    uint64_t reservation_id) const {
    if (reservation_id == 0) return true;
    std::lock_guard<std::mutex> lock(mutex_);
    return reservation_id == active_reservation_id_ &&
           !reserved_items_.empty() && !degraded_ && ready_ && accepting_ &&
           delivery_state_ == DeliveryState::kHealthy &&
           active_reservation_delivery_generation_ == delivery_generation_;
}

void SampleDistributor::CancelEnqueueEnvelopeSet(uint64_t reservation_id) {
    if (reservation_id == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (reservation_id != active_reservation_id_) return;
    reserved_items_.clear();
    reserved_transitions_ = 0;
    reserved_estimated_bytes_ = 0;
    active_reservation_id_ = 0;
    active_reservation_delivery_generation_ = 0;
    active_reservation_sealed_ = false;
    space_cv_.notify_all();
}

SampleDistributor::SendResult SampleDistributor::SendFront(
    const QueueItem& item,
    bool& duplicate,
    int& attempts_used,
    int& retry_after_ms,
    std::string& error) {
    duplicate = false;
    attempts_used = 0;
    retry_after_ms = 0;
    std::string pinned_sample_pool_instance_id;
    uint64_t pinned_sample_pool_lifecycle_epoch = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pinned_sample_pool_instance_id = sample_pool_instance_id_;
        pinned_sample_pool_lifecycle_epoch = sample_pool_lifecycle_epoch_;
    }

    for (int attempt = 0; attempt < config_.max_attempts; ++attempt) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (force_stop_) {
                error = "distributor forced to stop";
                return SendResult::kRejected;
            }
            ++push_attempt_count_;
            ++attempts_used;
            if (item.attempts + attempt > 0) ++retry_attempt_count_;
            if (!queue_.empty() &&
                queue_.front().envelope.envelope_id() ==
                    item.envelope.envelope_id()) {
                queue_.front().push_outcome_unknown = true;
            }
        }

        training::PushSamplesReq request;
        *request.mutable_envelope() = item.envelope;
        training::PushSamplesRsp response;
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() +
                             std::chrono::milliseconds(config_.rpc_timeout_ms));
        {
            std::lock_guard<std::mutex> rpc_lock(rpc_mutex_);
            active_rpc_ = &context;
        }
        const auto start = std::chrono::steady_clock::now();
        const grpc::Status status =
            stub_->PushSamples(&context, request, &response);
        const double latency_ms = ElapsedMs(start);
        {
            std::lock_guard<std::mutex> rpc_lock(rpc_mutex_);
            active_rpc_ = nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++push_rpc_count_;
            push_rpc_latency_sum_ms_ += latency_ms;
            push_rpc_latency_max_ms_ =
                std::max(push_rpc_latency_max_ms_, latency_ms);
        }

        if (status.ok()) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!queue_.empty() &&
                    queue_.front().envelope.envelope_id() ==
                        item.envelope.envelope_id()) {
                    queue_.front().push_outcome_unknown = false;
                }
            }
            if (!ValidatePushResponse(
                    item.envelope, item.transitions, response,
                    pinned_sample_pool_instance_id,
                    pinned_sample_pool_lifecycle_epoch, error)) {
                std::lock_guard<std::mutex> lock(mutex_);
                ++rejected_push_attempt_count_;
                return SendResult::kRejected;
            }
            if (response.result() == training::PUSH_RESULT_ACCEPTED ||
                response.result() == training::PUSH_RESULT_DUPLICATE) {
                duplicate =
                    response.result() == training::PUSH_RESULT_DUPLICATE;
                MarkHealthy();
                return SendResult::kCommitted;
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++rejected_push_attempt_count_;
            }
            error = response.message().empty()
                        ? "SamplePool rejected the immutable envelope"
                        : response.message();
            if (response.result() ==
                training::PUSH_RESULT_REJECTED_CAPACITY) {
                retry_after_ms = RetryBackoffMs(item.attempts + attempts_used);
                return SendResult::kTransient;
            }
            return SendResult::kRejected;
        }

        error = "PushSamples failed: " + status.error_message();
        if (!IsRetryableOutcomeUnknownTransport(status)) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!queue_.empty() &&
                queue_.front().envelope.envelope_id() ==
                    item.envelope.envelope_id()) {
                queue_.front().push_outcome_unknown = false;
            }
            ++rejected_push_attempt_count_;
            return SendResult::kRejected;
        }
        if (attempt + 1 >= config_.max_attempts) {
            retry_after_ms = RetryBackoffMs(item.attempts + attempts_used);
            return SendResult::kTransient;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(50 * (attempt + 1)));
    }
    return SendResult::kRejected;
}

void SampleDistributor::SenderLoop() {
    while (true) {
        QueueItem item;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            const bool signaled = queue_cv_.wait_for(
                lock,
                std::chrono::milliseconds(config_.status_poll_interval_ms),
                [this]() {
                    return force_stop_ || !queue_.empty() || stop_requested_;
                });
            if (force_stop_ || (stop_requested_ && queue_.empty())) break;
            if (!signaled && queue_.empty()) {
                lock.unlock();
                const auto refresh = RefreshSamplePoolStatus();
                if (refresh == StatusRefreshResult::kTerminal) break;
                if (refresh == StatusRefreshResult::kTransient) {
                    std::unique_lock<std::mutex> retry_lock(mutex_);
                    queue_cv_.wait_for(
                        retry_lock,
                        std::chrono::milliseconds(transient_retry_after_ms_),
                        [this]() {
                            return force_stop_ || stop_requested_ ||
                                   !queue_.empty();
                        });
                }
                continue;
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
                queue_.front().envelope.envelope_id() ==
                    item.envelope.envelope_id()) {
                queue_transitions_ -= queue_.front().transitions;
                queue_estimated_bytes_ -= queue_.front().estimated_bytes;
                queue_.pop_front();
                if (duplicate) {
                    ++duplicate_push_attempt_count_;
                } else {
                    accepted_unique_transitions_ += item.transitions;
                    ++accepted_unique_envelopes_;
                }
                space_cv_.notify_all();
                if (queue_.empty()) drained_cv_.notify_all();
            }
            continue;
        }

        if (result == SendResult::kTransient) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!queue_.empty() &&
                    queue_.front().envelope.envelope_id() ==
                        item.envelope.envelope_id()) {
                    queue_.front().attempts += attempts_used;
                }
                if (force_stop_) break;
            }
            if (!MarkTransient(
                    error.empty()
                        ? "SamplePool ingress transport is unavailable"
                        : error,
                    retry_after_ms)) {
                break;
            }
            std::unique_lock<std::mutex> lock(mutex_);
            queue_cv_.wait_for(
                lock, std::chrono::milliseconds(transient_retry_after_ms_),
                [this]() { return force_stop_; });
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!queue_.empty() &&
                queue_.front().envelope.envelope_id() ==
                    item.envelope.envelope_id()) {
                queue_.front().attempts += attempts_used;
            }
        }
        MarkDegraded(error.empty() ? "PushSamples failed" : error);
        break;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    ready_ = false;
    if (delivery_state_ != DeliveryState::kTerminalFault) {
        SetDeliveryStateLocked(DeliveryState::kStopped);
    }
    drained_cv_.notify_all();
}

void SampleDistributor::CancelActiveRpc() {
    std::lock_guard<std::mutex> lock(rpc_mutex_);
    if (active_rpc_) active_rpc_->TryCancel();
}

bool SampleDistributor::StopAndDrain() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        reserved_items_.clear();
        reserved_transitions_ = 0;
        reserved_estimated_bytes_ = 0;
        active_reservation_id_ = 0;
        active_reservation_delivery_generation_ = 0;
        active_reservation_sealed_ = false;
        if (!sender_thread_.joinable()) {
            accepting_ = false;
            ready_ = false;
            if (delivery_state_ != DeliveryState::kTerminalFault) {
                SetDeliveryStateLocked(DeliveryState::kStopped);
            }
            return queue_.empty();
        }
        accepting_ = false;
        stop_requested_ = true;
        if (delivery_state_ != DeliveryState::kTerminalFault) {
            SetDeliveryStateLocked(DeliveryState::kDraining);
        }
        queue_cv_.notify_all();
    }

    bool drained = false;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        drained = drained_cv_.wait_for(
            lock, std::chrono::milliseconds(config_.drain_timeout_ms),
            [this]() { return queue_.empty(); });
        if (!drained) {
            force_stop_ = true;
            queue_cv_.notify_all();
            space_cv_.notify_all();
        }
    }
    if (!drained) CancelActiveRpc();
    if (sender_thread_.joinable()) sender_thread_.join();
    if (!drained) {
        std::lock_guard<std::mutex> lock(mutex_);
        drained = queue_.empty();
        if (!drained) {
            for (const auto& item : queue_) {
                if (item.push_outcome_unknown) {
                    unresolved_push_outcome_unknown_transitions_ +=
                        item.transitions;
                    ++unresolved_push_outcome_unknown_envelopes_;
                } else {
                    final_drop_unique_transitions_ += item.transitions;
                    ++final_drop_unique_envelopes_;
                }
            }
            queue_.clear();
            queue_transitions_ = 0;
            queue_estimated_bytes_ = 0;
            degraded_ = true;
            SetDeliveryStateLocked(DeliveryState::kTerminalFault);
            last_error_ = unresolved_push_outcome_unknown_envelopes_ > 0
                              ? "outbound drain deadline exceeded with "
                                "unresolved PushSamples outcomes"
                              : "outbound drain deadline exceeded";
        }
    }
    return drained;
}

bool SampleDistributor::IsReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ready_;
}

bool SampleDistributor::IsDegraded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return degraded_;
}

bool SampleDistributor::IsTrainingDeliveryReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return accepting_ && ready_ && !degraded_ &&
           delivery_state_ == DeliveryState::kHealthy &&
           !ProducerCapacityConstrainedLocked();
}

bool SampleDistributor::IsPausedAtSafeBoundary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return delivery_state_ == DeliveryState::kTransientRetry ||
           ProducerCapacityConstrainedLocked();
}

int SampleDistributor::PauseRetryAfterMs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (transient_retry_after_ms_ > 0) return transient_retry_after_ms_;
    return config_.enqueue_timeout_ms;
}

bool SampleDistributor::ProducerCapacityConstrainedLocked() const {
    return queue_.size() + reserved_items_.size() >=
               config_.outbound_max_envelopes ||
           queue_estimated_bytes_ + reserved_estimated_bytes_ >=
               static_cast<int64_t>(
                   config_.outbound_max_estimated_bytes);
}

void SampleDistributor::SetDeliveryStateLocked(DeliveryState state) {
    if (delivery_state_ == state) return;
    delivery_state_ = state;
    ++delivery_generation_;
    if (delivery_generation_ == 0) ++delivery_generation_;
}

void SampleDistributor::MarkDegraded(const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    degraded_ = true;
    ready_ = false;
    recovery_active_ = false;
    SetDeliveryStateLocked(DeliveryState::kTerminalFault);
    transient_retry_after_ms_ = 0;
    last_error_ = error;
    space_cv_.notify_all();
}

bool SampleDistributor::MarkTransient(const std::string& error,
                                      int retry_after_ms) {
    bool entered_retry = false;
    bool deadline_exceeded = false;
    int64_t elapsed_ms = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (degraded_ || delivery_state_ == DeliveryState::kTerminalFault ||
            delivery_state_ == DeliveryState::kDraining ||
            delivery_state_ == DeliveryState::kStopped) {
            return false;
        }
        const auto now = std::chrono::steady_clock::now();
        if (!recovery_active_) {
            recovery_active_ = true;
            recovery_started_ = now;
            entered_retry = true;
        }
        elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         now - recovery_started_)
                         .count();
        deadline_exceeded = elapsed_ms >=
                            static_cast<int64_t>(
                                config_.recovery_timeout_ms);
        ready_ = false;
        if (deadline_exceeded) {
            degraded_ = true;
            recovery_active_ = false;
            SetDeliveryStateLocked(DeliveryState::kTerminalFault);
            transient_retry_after_ms_ = 0;
            last_error_ =
                "SamplePool ingress recovery deadline exceeded after " +
                std::to_string(elapsed_ms) + "ms: " + error;
        } else {
            SetDeliveryStateLocked(DeliveryState::kTransientRetry);
            transient_retry_after_ms_ = retry_after_ms;
            last_error_ = error;
        }
        space_cv_.notify_all();
    }
    if (deadline_exceeded) {
        LOG_ERROR("SampleDistributor",
                  "SamplePool ingress 恢复期限已耗尽: elapsed_ms=%lld error=%s",
                  static_cast<long long>(elapsed_ms), error.c_str());
        return false;
    }
    if (entered_retry) {
        LOG_WARN("SampleDistributor",
                 "SamplePool ingress 进入有界恢复: retry_after_ms=%d recovery_timeout_ms=%d error=%s",
                 retry_after_ms, config_.recovery_timeout_ms,
                 error.c_str());
    }
    return true;
}

void SampleDistributor::MarkHealthy() {
    bool recovered = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (degraded_ || delivery_state_ == DeliveryState::kTerminalFault ||
            delivery_state_ == DeliveryState::kDraining ||
            delivery_state_ == DeliveryState::kStopped) {
            return;
        }
        recovered = delivery_state_ == DeliveryState::kTransientRetry;
        status_failure_attempts_ = 0;
        ready_ = true;
        recovery_active_ = false;
        SetDeliveryStateLocked(DeliveryState::kHealthy);
        transient_retry_after_ms_ = 0;
        last_error_.clear();
    }
    if (recovered) {
        LOG_INFO("SampleDistributor", "SamplePool ingress 瞬态故障已恢复");
    }
}

void SampleDistributor::RecordFinalDrop(int64_t transitions,
                                        int64_t envelopes,
                                        const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    final_drop_unique_transitions_ += transitions;
    final_drop_unique_envelopes_ += envelopes;
    degraded_ = true;
    SetDeliveryStateLocked(DeliveryState::kTerminalFault);
    last_error_ = error;
}

SampleDistributor::Snapshot SampleDistributor::GetSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Snapshot snapshot;
    snapshot.delivery_state = delivery_state_;
    snapshot.ready = ready_;
    snapshot.degraded = degraded_;
    snapshot.transient_retry =
        delivery_state_ == DeliveryState::kTransientRetry;
    snapshot.backend_recovering = snapshot.transient_retry;
    snapshot.terminal_fault =
        delivery_state_ == DeliveryState::kTerminalFault;
    snapshot.local_backpressure = ProducerCapacityConstrainedLocked();
    snapshot.sample_delivery_paused =
        snapshot.transient_retry || snapshot.local_backpressure;
    if (recovery_active_) {
        snapshot.recovery_elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - recovery_started_)
                .count();
    }
    snapshot.retry_after_ms = transient_retry_after_ms_;
    snapshot.queue_envelopes = queue_.size();
    snapshot.queue_transitions = queue_transitions_;
    snapshot.queue_estimated_bytes = queue_estimated_bytes_;
    snapshot.queue_high_watermark = queue_high_watermark_;
    snapshot.push_attempt_count = push_attempt_count_;
    snapshot.accepted_unique_transitions =
        accepted_unique_transitions_;
    snapshot.accepted_unique_envelopes = accepted_unique_envelopes_;
    snapshot.duplicate_push_attempt_count = duplicate_push_attempt_count_;
    snapshot.rejected_push_attempt_count = rejected_push_attempt_count_;
    snapshot.retry_attempt_count = retry_attempt_count_;
    snapshot.final_drop_unique_transitions =
        final_drop_unique_transitions_;
    snapshot.final_drop_unique_envelopes = final_drop_unique_envelopes_;
    snapshot.unresolved_push_outcome_unknown_transitions =
        unresolved_push_outcome_unknown_transitions_;
    snapshot.unresolved_push_outcome_unknown_envelopes =
        unresolved_push_outcome_unknown_envelopes_;
    snapshot.push_rpc_count = push_rpc_count_;
    snapshot.push_rpc_latency_sum_ms = push_rpc_latency_sum_ms_;
    snapshot.push_rpc_latency_max_ms = push_rpc_latency_max_ms_;
    snapshot.sample_pool_instance_id = sample_pool_instance_id_;
    snapshot.last_error = last_error_;
    return snapshot;
}
