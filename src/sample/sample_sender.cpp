#include "sample/sample_sender.h"

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

int64_t UnixNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

bool SameDigest(const common::ContentDigest& lhs,
                const common::ContentDigest& rhs) {
    return lhs.algorithm() == rhs.algorithm() && lhs.hex() == rhs.hex();
}

bool SameDistributor(
    const common::ServiceInstanceIdentity& actual,
    const std::string& expected_instance_id,
    uint64_t expected_lifecycle_epoch) {
    return actual.component() == "sample-distributor" &&
           actual.instance_id() == expected_instance_id &&
           actual.lifecycle_epoch() == expected_lifecycle_epoch;
}

bool IsCreditWait(training::SampleCreditResult result) {
    return result == training::SAMPLE_CREDIT_RESULT_WAIT_NO_DEMAND ||
           result == training::SAMPLE_CREDIT_RESULT_WAIT_INFLIGHT_LIMIT ||
           result == training::SAMPLE_CREDIT_RESULT_WAIT_CAPACITY ||
           result == training::SAMPLE_CREDIT_RESULT_WAIT_DRAINING;
}

bool IsCreditRejection(training::SampleCreditResult result) {
    return result == training::SAMPLE_CREDIT_RESULT_REJECTED_IDENTITY ||
           result == training::SAMPLE_CREDIT_RESULT_REJECTED_SEMANTICS ||
           result == training::SAMPLE_CREDIT_RESULT_REJECTED_FRESHNESS ||
           result == training::SAMPLE_CREDIT_RESULT_REJECTED_INVALID;
}

bool HasCreditAuthorizationPayload(
    const training::SampleCreditGrant& response) {
    return !response.credit_id().empty() || !response.demand_id().empty() ||
           response.demand_epoch() != 0 ||
           response.payload_digest().algorithm() !=
               common::DIGEST_ALGORITHM_UNSPECIFIED ||
           !response.payload_digest().hex().empty() ||
           response.granted_samples() != 0 ||
           response.granted_fragments() != 0 ||
           response.granted_estimated_bytes() != 0 ||
           response.expires_at_unix_ms() != 0 ||
           response.state() != training::SAMPLE_CREDIT_STATE_UNSPECIFIED;
}

bool ValidateCreditResponse(
    const training::AcquireSampleCreditReq& request,
    const training::SampleCreditGrant& response,
    const std::string& distributor_instance_id,
    uint64_t distributor_lifecycle_epoch,
    std::string& error) {
    if (!SameDistributor(response.distributor(), distributor_instance_id,
                         distributor_lifecycle_epoch)) {
        error = "sample credit response distributor identity changed";
        return false;
    }
    if (response.request_id() != request.request_id() ||
        response.batch_id() != request.batch_id()) {
        error = "sample credit response does not echo the requested batch";
        return false;
    }

    if (response.result() == training::SAMPLE_CREDIT_RESULT_GRANTED) {
        if (response.ret_code() != 0 || response.credit_id().empty() ||
            response.demand_id().empty() || response.demand_epoch() == 0 ||
            !SameDigest(response.payload_digest(), request.payload_digest()) ||
            response.granted_samples() != request.sample_count() ||
            response.granted_fragments() != request.fragment_count() ||
            response.granted_estimated_bytes() != request.estimated_bytes()) {
            error = "sample credit grant does not authorize the exact batch";
            return false;
        }
        if (response.state() == training::SAMPLE_CREDIT_STATE_RESERVED) {
            if (response.expires_at_unix_ms() <= UnixNowMs()) {
                error = "sample credit grant is already expired";
                return false;
            }
        } else if (response.state() !=
                   training::SAMPLE_CREDIT_STATE_COMMITTED) {
            error = "sample credit grant is not reserved or committed";
            return false;
        }
        // COMMITTED is the idempotent recovery response after the Pool may
        // have applied a Push whose reply was lost. Its original reservation
        // expiry may be in the past; the exact duplicate Push below is still
        // required before the local batch becomes disposable.
        return true;
    }
    if (IsCreditWait(response.result())) {
        if (response.ret_code() <= 0 ||
            HasCreditAuthorizationPayload(response)) {
            error =
                "sample credit WAIT result has contradictory grant payload";
            return false;
        }
        return true;
    }
    if (IsCreditRejection(response.result())) {
        if (response.ret_code() >= 0 ||
            HasCreditAuthorizationPayload(response)) {
            error =
                "sample credit rejection has contradictory grant payload";
            return false;
        }
        return true;
    }
    error = "sample credit response result is unspecified";
    return false;
}

bool IsPushRejection(training::PushResult result) {
    return result == training::PUSH_RESULT_REJECTED_CAPACITY ||
           result == training::PUSH_RESULT_REJECTED_INVALID ||
           result == training::PUSH_RESULT_REJECTED_IDENTITY;
}

bool IsRetryableOutcomeUnknownTransport(const grpc::Status& status) {
    // A non-OK unary result does not prove whether the remote handler committed
    // before the transport failed. These codes are retryable with the exact
    // immutable request/batch identity; application/contract errors still fail
    // closed after an OK response is validated.
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
    const training::SampleBatch& batch,
    int64_t sample_count,
    const training::PushSamplesRsp& response,
    const std::string& distributor_instance_id,
    uint64_t distributor_lifecycle_epoch,
    std::string& error) {
    if (!SameDistributor(response.distributor(), distributor_instance_id,
                         distributor_lifecycle_epoch)) {
        error = "PushSamples response distributor identity changed";
        return false;
    }
    if (response.batch_id() != batch.batch_id()) {
        error = "PushSamples response does not echo the exact batch_id";
        return false;
    }
    if (response.result() == training::PUSH_RESULT_ACCEPTED) {
        if (response.ret_code() != 0 ||
            response.accepted_samples() != sample_count ||
            response.accepted_unique_samples() != sample_count) {
            error = "PushSamples ACCEPTED response has inconsistent counts";
            return false;
        }
        return true;
    }
    if (response.result() == training::PUSH_RESULT_DUPLICATE) {
        if (response.ret_code() != 0 || response.accepted_samples() != 0 ||
            response.accepted_unique_samples() != 0) {
            error = "PushSamples DUPLICATE response claims new acceptance";
            return false;
        }
        return true;
    }
    if (IsPushRejection(response.result())) {
        if (response.ret_code() == 0 || response.accepted_samples() != 0 ||
            response.accepted_unique_samples() != 0) {
            error = "PushSamples rejection has inconsistent status or counts";
            return false;
        }
        return true;
    }
    error = "PushSamples response result is unspecified";
    return false;
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
    std::string error;
    if (!ValidateDistributorStatus(response, error) ||
        !ApplyDistributorStatus(response, true, error)) {
        MarkDegraded(error);
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    distributor_instance_id_ = response.distributor().instance_id();
    distributor_lifecycle_epoch_ = response.distributor().lifecycle_epoch();
    ready_ = true;
    degraded_ = false;
    SetDeliveryStateLocked(DeliveryState::kHealthy);
    transient_retry_after_ms_ = 0;
    last_error_.clear();
    return true;
}

bool SampleSender::ValidateDistributorStatus(
    const training::DistributorStatusRsp& response,
    std::string& error) const {
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
        error = "sample ingress identity does not match rl-contracts 0.11.0";
        return false;
    }
    if (!response.ready() || !response.ingress_ready()) {
        error = "sample ingress is not ready";
        return false;
    }
    return true;
}

bool SampleSender::ApplyDistributorStatus(
    const training::DistributorStatusRsp& response,
    bool initialize_baseline,
    std::string& error) {
    std::unordered_map<ModelVersion, int64_t> current_by_model;
    int64_t current_sum = 0;
    for (const auto& status : response.behavior_versions()) {
        const int64_t stale = status.stale_samples();
        if (stale < 0 ||
            stale > std::numeric_limits<int64_t>::max() - current_sum) {
            error = "sample ingress returned invalid stale accounting";
            return false;
        }
        const ModelVersion version =
            status.behavior_policy().model_version();
        if (current_by_model[version] >
            std::numeric_limits<int64_t>::max() - stale) {
            error = "sample ingress stale accounting overflowed";
            return false;
        }
        current_by_model[version] += stale;
        current_sum += stale;
    }
    if (response.stale_sample_count() < 0 ||
        current_sum != response.stale_sample_count()) {
        error = "sample ingress stale accounting does not balance";
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (initialize_baseline) {
        pool_stale_baseline_count_ = response.stale_sample_count();
        pool_stale_baseline_by_model_ = std::move(current_by_model);
        pool_stale_count_ = 0;
        pool_stale_samples_by_model_.clear();
        return true;
    }
    if (response.distributor().instance_id() != distributor_instance_id_ ||
        response.distributor().lifecycle_epoch() !=
            distributor_lifecycle_epoch_) {
        error = "sample ingress identity changed during training";
        return false;
    }
    if (response.stale_sample_count() < pool_stale_baseline_count_) {
        error = "sample ingress stale counter moved backwards";
        return false;
    }

    std::unordered_map<ModelVersion, int64_t> delta_by_model;
    int64_t delta_sum = 0;
    for (const auto& baseline : pool_stale_baseline_by_model_) {
        const auto current = current_by_model.find(baseline.first);
        if (current == current_by_model.end() ||
            current->second < baseline.second) {
            error = "sample ingress behavior stale counter moved backwards";
            return false;
        }
    }
    for (const auto& current : current_by_model) {
        const auto baseline = pool_stale_baseline_by_model_.find(current.first);
        const int64_t baseline_value =
            baseline == pool_stale_baseline_by_model_.end()
                ? 0
                : baseline->second;
        const int64_t delta = current.second - baseline_value;
        if (delta > 0) delta_by_model.emplace(current.first, delta);
        delta_sum += delta;
    }
    if (delta_sum !=
        response.stale_sample_count() - pool_stale_baseline_count_) {
        error = "sample ingress stale delta does not balance";
        return false;
    }
    pool_stale_count_ = delta_sum;
    pool_stale_samples_by_model_ = std::move(delta_by_model);
    return true;
}

SampleSender::StatusRefreshResult SampleSender::RefreshDistributorStatus() {
    training::DistributorStatusReq request;
    training::DistributorStatusRsp response;
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
        const std::string error =
            "sample ingress status refresh failed: " +
            status.error_message();
        if (IsRetryableOutcomeUnknownTransport(status)) {
            int attempts = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                status_failure_attempts_ =
                    std::min(6, status_failure_attempts_ + 1);
                attempts = status_failure_attempts_;
            }
            MarkTransient(error, RetryBackoffMs(attempts));
            return StatusRefreshResult::kTransient;
        }
        MarkDegraded(error);
        return StatusRefreshResult::kTerminal;
    }
    std::string error;
    if (!ValidateDistributorStatus(response, error) ||
        !ApplyDistributorStatus(response, false, error)) {
        MarkDegraded(error);
        return StatusRefreshResult::kTerminal;
    }
    MarkHealthy();
    return StatusRefreshResult::kHealthy;
}

bool SampleSender::Start() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        SetDeliveryStateLocked(DeliveryState::kStarting);
    }
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
        transient_retry_after_ms_ = 0;
        status_failure_attempts_ = 0;
        SetDeliveryStateLocked(DeliveryState::kHealthy);
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
               (queue_.size() + reserved_items_.size() <
                    config_.outbound_max_fragments &&
                queue_estimated_bytes_ + reserved_estimated_bytes_ +
                        item.estimated_bytes <=
                    static_cast<int64_t>(config_.outbound_max_estimated_bytes));
    };
    if (!space_cv_.wait_for(lock,
                            std::chrono::milliseconds(config_.enqueue_timeout_ms),
                            has_capacity)) {
        degraded_ = true;
        SetDeliveryStateLocked(DeliveryState::kTerminalFault);
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

SampleSender::ReservationResult SampleSender::ReserveEnqueueBatchSet(
    const std::vector<training::SampleBatch>& batches,
    uint64_t& reservation_id,
    std::string& error) {
    reservation_id = 0;
    error.clear();
    if (batches.empty()) return ReservationResult::kReserved;

    std::list<QueueItem> items;
    int64_t samples = 0;
    int64_t estimated_bytes = 0;
    for (const auto& batch : batches) {
        QueueItem item;
        item.batch = batch;
        item.samples = batch.samples_size();
        item.estimated_bytes = static_cast<int64_t>(batch.ByteSizeLong());
        if (item.samples <= 0 || item.estimated_bytes <= 0 ||
            samples > std::numeric_limits<int64_t>::max() - item.samples ||
            estimated_bytes >
                std::numeric_limits<int64_t>::max() - item.estimated_bytes) {
            error = "sample batch reservation is invalid";
            return ReservationResult::kTerminalFault;
        }
        samples += item.samples;
        estimated_bytes += item.estimated_bytes;
        items.push_back(std::move(item));
    }

    std::unique_lock<std::mutex> lock(mutex_);
    auto has_capacity = [this, &items, estimated_bytes]() {
        return force_stop_ || !accepting_ || degraded_ ||
               delivery_state_ != DeliveryState::kHealthy ||
               (reserved_items_.empty() &&
                queue_.size() + items.size() <=
                    config_.outbound_max_fragments &&
                queue_estimated_bytes_ + estimated_bytes <=
                    static_cast<int64_t>(
                        config_.outbound_max_estimated_bytes));
    };
    if (!space_cv_.wait_for(
            lock, std::chrono::milliseconds(config_.enqueue_timeout_ms),
            has_capacity)) {
        error = "outbound queue batch-set reservation timeout";
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
                    : "sample delivery is waiting for flow control";
        return ReservationResult::kRetryableUnavailable;
    }
    if (!reserved_items_.empty()) {
        error = "another outbound batch-set reservation is active";
        return ReservationResult::kTerminalFault;
    }

    active_reservation_id_ = next_reservation_id_++;
    if (active_reservation_id_ == 0) {
        active_reservation_id_ = next_reservation_id_++;
    }
    reserved_items_.splice(reserved_items_.end(), items);
    reserved_samples_ = samples;
    reserved_estimated_bytes_ = estimated_bytes;
    active_reservation_delivery_generation_ = delivery_generation_;
    active_reservation_sealed_ = false;
    reservation_id = active_reservation_id_;
    return ReservationResult::kReserved;
}

SampleSender::SealResult SampleSender::SealEnqueueBatchSet(
    uint64_t reservation_id, std::string& error) {
    error.clear();
    if (reservation_id == 0) return SealResult::kSealed;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto release_reservation = [this]() {
        reserved_items_.clear();
        reserved_samples_ = 0;
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

SampleSender::CommitResult SampleSender::CommitEnqueueBatchSet(
    uint64_t reservation_id, std::string& error) {
    error.clear();
    if (reservation_id == 0) return CommitResult::kCommitted;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto release_reservation = [this]() {
        reserved_items_.clear();
        reserved_samples_ = 0;
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
        // Seal is the local point of no return. Model publication may happen
        // after it, so delivery faults can delay sending but cannot reject or
        // partially expose the already-authorized frame transaction.
        queue_.splice(queue_.end(), reserved_items_);
        queue_samples_ += reserved_samples_;
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
    // std::list::splice performs no allocation and publishes the whole
    // preallocated group while holding the sender mutex.
    queue_.splice(queue_.end(), reserved_items_);
    queue_samples_ += reserved_samples_;
    queue_estimated_bytes_ += reserved_estimated_bytes_;
    queue_high_watermark_ =
        std::max(queue_high_watermark_, static_cast<int64_t>(queue_.size()));
    reserved_items_.clear();
    reserved_samples_ = 0;
    reserved_estimated_bytes_ = 0;
    active_reservation_id_ = 0;
    active_reservation_delivery_generation_ = 0;
    active_reservation_sealed_ = false;
    queue_cv_.notify_one();
    return CommitResult::kCommitted;
}

bool SampleSender::HasEnqueueReservation(uint64_t reservation_id) const {
    if (reservation_id == 0) return true;
    std::lock_guard<std::mutex> lock(mutex_);
    return reservation_id == active_reservation_id_ &&
           !reserved_items_.empty() && !degraded_ && ready_ && accepting_ &&
           delivery_state_ == DeliveryState::kHealthy &&
           active_reservation_delivery_generation_ == delivery_generation_;
}

void SampleSender::CancelEnqueueBatchSet(uint64_t reservation_id) {
    if (reservation_id == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (reservation_id != active_reservation_id_) return;
    reserved_items_.clear();
    reserved_samples_ = 0;
    reserved_estimated_bytes_ = 0;
    active_reservation_id_ = 0;
    active_reservation_delivery_generation_ = 0;
    active_reservation_sealed_ = false;
    space_cv_.notify_all();
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
    std::string pinned_distributor_instance_id;
    uint64_t pinned_distributor_lifecycle_epoch = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pinned_distributor_instance_id = distributor_instance_id_;
        pinned_distributor_lifecycle_epoch = distributor_lifecycle_epoch_;
    }

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
            error = "AcquireSampleCredit failed: " +
                    credit_status.error_message();
            if (attempt + 1 < attempts) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(50 * (attempt + 1)));
                continue;
            }
            if (IsRetryableOutcomeUnknownTransport(credit_status)) {
                retry_after_ms = RetryBackoffMs(item.attempts + attempts_used);
                return SendResult::kTransient;
            }
            return SendResult::kRejected;
        }
        if (!ValidateCreditResponse(
                credit_request, credit, pinned_distributor_instance_id,
                pinned_distributor_lifecycle_epoch, error)) {
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
                    SetDeliveryStateLocked(DeliveryState::kFlowWait);
                    transient_retry_after_ms_ = 0;
                    status_failure_attempts_ = 0;
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
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!queue_.empty() &&
                queue_.front().batch.batch_id() == item.batch.batch_id()) {
                // From the instant a Push may leave this process until a
                // validated response resolves it, shutdown must conservatively
                // treat the remote disposition as unknown.
                queue_.front().push_outcome_unknown = true;
            }
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

        if (status.ok()) {
            if (!ValidatePushResponse(
                    item.batch, item.samples, response,
                    pinned_distributor_instance_id,
                    pinned_distributor_lifecycle_epoch, error)) {
                if (attempt + 1 < attempts) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(50 * (attempt + 1)));
                    continue;
                }
                // An OK transport with an invalid protobuf still cannot prove
                // whether the handler committed before producing the malformed
                // reply. Keep the immutable front and converge by exact retry;
                // only an OK *validated* application rejection is conclusive.
                retry_after_ms = RetryBackoffMs(
                    item.attempts + attempts_used);
                return SendResult::kTransient;
            }
            if (response.result() == training::PUSH_RESULT_ACCEPTED ||
                response.result() == training::PUSH_RESULT_DUPLICATE) {
                duplicate =
                    response.result() == training::PUSH_RESULT_DUPLICATE;
                bool recovered = false;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (training_capacity_wait_) {
                        capacity_wait_ms_ +=
                            std::chrono::duration_cast<
                                std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() -
                                capacity_wait_started_)
                                .count();
                        training_capacity_wait_ = false;
                        training_capacity_retry_after_ms_ = 0;
                    }
                    recovered =
                        delivery_state_ == DeliveryState::kTransientRetry;
                    SetDeliveryStateLocked(DeliveryState::kHealthy);
                    transient_retry_after_ms_ = 0;
                    status_failure_attempts_ = 0;
                    last_error_.clear();
                }
                if (recovered) {
                    LOG_INFO("SampleSender",
                             "样本链路瞬态故障已恢复: batch_id=%s",
                             item.batch.batch_id().c_str());
                }
                return SendResult::kCommitted;
            }
            error = response.message();
            std::lock_guard<std::mutex> lock(mutex_);
            if (!queue_.empty() &&
                queue_.front().batch.batch_id() == item.batch.batch_id()) {
                queue_.front().push_outcome_unknown = false;
            }
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
            error = "PushSamples failed: " + status.error_message();
            // Once PushSamples was sent, every non-OK unary completion is
            // outcome-unknown: the remote handler may already have committed
            // the immutable batch before an intermediary rewrote the status.
            // Only an OK, validated protobuf rejection is conclusive.
            if (attempt + 1 >= attempts) {
                retry_after_ms = RetryBackoffMs(item.attempts + attempts_used);
                return SendResult::kTransient;
            }
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
            const bool signaled = queue_cv_.wait_for(
                lock,
                std::chrono::milliseconds(config_.status_poll_interval_ms),
                [this]() {
                    return force_stop_ || !queue_.empty() || stop_requested_;
                });
            if (force_stop_ || (stop_requested_ && queue_.empty())) {
                break;
            }
            if (!signaled && queue_.empty()) {
                lock.unlock();
                const auto refresh = RefreshDistributorStatus();
                if (refresh == StatusRefreshResult::kTerminal) break;
                if (refresh == StatusRefreshResult::kTransient) {
                    std::unique_lock<std::mutex> retry_lock(mutex_);
                    queue_cv_.wait_for(
                        retry_lock,
                        std::chrono::milliseconds(
                            std::max(1, transient_retry_after_ms_)),
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
                queue_.front().batch.batch_id() == item.batch.batch_id()) {
                queue_samples_ -= queue_.front().samples;
                queue_estimated_bytes_ -= queue_.front().estimated_bytes;
                queue_.pop_front();
                // Both validated outcomes prove that this producer's exact
                // deterministic queue front is present in the Pool. DUPLICATE
                // diagnoses response-loss recovery; it does not undo the
                // unique accepted disposition of the local produced batch.
                accepted_unique_samples_ += item.samples;
                ++accepted_unique_batches_;
                if (duplicate) ++duplicate_push_attempt_count_;
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
                producer_stale_samples_by_model_[
                    queue_.front().batch.behavior_policy().model_version()] +=
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
                SetDeliveryStateLocked(DeliveryState::kHealthy);
                transient_retry_after_ms_ = 0;
                status_failure_attempts_ = 0;
                last_error_.clear();
                space_cv_.notify_all();
                if (queue_.empty()) drained_cv_.notify_all();
            }
            continue;
        }

        if (result == SendResult::kTransient) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!queue_.empty() &&
                    queue_.front().batch.batch_id() == item.batch.batch_id()) {
                    queue_.front().attempts += attempts_used;
                }
                if (force_stop_) break;
            }
            MarkTransient(
                error.empty() ? "sample ingress transport is unavailable"
                              : error,
                std::max(1, retry_after_ms));
            std::unique_lock<std::mutex> lock(mutex_);
            queue_cv_.wait_for(
                lock,
                std::chrono::milliseconds(
                    std::max(1, transient_retry_after_ms_)),
                [this]() { return force_stop_; });
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
        break;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    ready_ = false;
    if (delivery_state_ != DeliveryState::kTerminalFault) {
        SetDeliveryStateLocked(DeliveryState::kStopped);
    }
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
        reserved_items_.clear();
        reserved_samples_ = 0;
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
            // Let the sender thread resolve a response that has already
            // returned before classifying the front. Clearing here races the
            // validated ACCEPTED/DUPLICATE -> pop/account step and can turn a
            // known remote commit into a false unresolved disposition.
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
                    unresolved_push_outcome_unknown_samples_ += item.samples;
                    ++unresolved_push_outcome_unknown_batches_;
                } else {
                    final_drop_unique_samples_ += item.samples;
                    ++final_drop_unique_batches_;
                }
            }
            queue_.clear();
            queue_samples_ = 0;
            queue_estimated_bytes_ = 0;
            degraded_ = true;
            SetDeliveryStateLocked(DeliveryState::kTerminalFault);
            last_error_ = unresolved_push_outcome_unknown_batches_ > 0
                ? "outbound drain deadline exceeded with unresolved "
                  "PushSamples outcomes"
                : "outbound drain deadline exceeded";
        }
    }
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

bool SampleSender::IsTrainingDeliveryReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return accepting_ && ready_ && !degraded_ &&
           delivery_state_ == DeliveryState::kHealthy &&
           !ProducerCapacityConstrainedLocked();
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
    if (queue_.size() + reserved_items_.size() +
            producer_fragment_reserve_ >
        config_.outbound_max_fragments) {
        return true;
    }

    const auto bytes_per_slot =
        config_.outbound_max_estimated_bytes /
        config_.outbound_max_fragments;
    const auto byte_reserve =
        bytes_per_slot * producer_fragment_reserve_;
    return byte_reserve > config_.outbound_max_estimated_bytes ||
           queue_estimated_bytes_ + reserved_estimated_bytes_ >
               static_cast<int64_t>(
                   config_.outbound_max_estimated_bytes - byte_reserve);
}

void SampleSender::SetDeliveryStateLocked(DeliveryState state) {
    if (delivery_state_ == state) return;
    delivery_state_ = state;
    ++delivery_generation_;
    if (delivery_generation_ == 0) ++delivery_generation_;
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
    SetDeliveryStateLocked(DeliveryState::kTerminalFault);
    transient_retry_after_ms_ = 0;
    last_error_ = error;
    space_cv_.notify_all();
}

void SampleSender::MarkTransient(const std::string& error,
                                 int retry_after_ms) {
    bool entered_retry = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (degraded_ || delivery_state_ == DeliveryState::kTerminalFault ||
            delivery_state_ == DeliveryState::kDraining ||
            delivery_state_ == DeliveryState::kStopped) {
            return;
        }
        if (training_capacity_wait_) {
            capacity_wait_ms_ +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - capacity_wait_started_)
                    .count();
            training_capacity_wait_ = false;
            training_capacity_retry_after_ms_ = 0;
        }
        entered_retry =
            delivery_state_ != DeliveryState::kTransientRetry;
        SetDeliveryStateLocked(DeliveryState::kTransientRetry);
        transient_retry_after_ms_ = std::max(1, retry_after_ms);
        last_error_ = error;
        space_cv_.notify_all();
    }
    if (entered_retry) {
        LOG_WARN("SampleSender",
                 "样本链路进入瞬态恢复: retry_after_ms=%d error=%s",
                 std::max(1, retry_after_ms), error.c_str());
    }
}

void SampleSender::MarkHealthy() {
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
        SetDeliveryStateLocked(
            training_capacity_wait_ ? DeliveryState::kFlowWait
                                    : DeliveryState::kHealthy);
        transient_retry_after_ms_ = 0;
        last_error_.clear();
    }
    if (recovered) {
        LOG_INFO("SampleSender", "样本链路瞬态故障已恢复");
    }
}

void SampleSender::RecordFinalDrop(int64_t samples, int64_t batches,
                                   const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    final_drop_unique_samples_ += samples;
    final_drop_unique_batches_ += batches;
    degraded_ = true;
    SetDeliveryStateLocked(DeliveryState::kTerminalFault);
    last_error_ = error;
}

SampleSender::Snapshot SampleSender::GetSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Snapshot snapshot;
    snapshot.delivery_state = delivery_state_;
    snapshot.ready = ready_;
    snapshot.degraded = degraded_;
    snapshot.transient_retry =
        delivery_state_ == DeliveryState::kTransientRetry;
    snapshot.terminal_fault =
        delivery_state_ == DeliveryState::kTerminalFault;
    snapshot.training_capacity_wait = ProducerCapacityConstrainedLocked();
    if (transient_retry_after_ms_ > 0) {
        snapshot.retry_after_ms = transient_retry_after_ms_;
    } else if (training_capacity_retry_after_ms_ > 0) {
        snapshot.retry_after_ms = training_capacity_retry_after_ms_;
    } else {
        snapshot.retry_after_ms = std::max(1, config_.enqueue_timeout_ms);
    }
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
    snapshot.unresolved_push_outcome_unknown_samples =
        unresolved_push_outcome_unknown_samples_;
    snapshot.unresolved_push_outcome_unknown_batches =
        unresolved_push_outcome_unknown_batches_;
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
    snapshot.pool_stale_count = pool_stale_count_;
    snapshot.pool_stale_samples_by_model = pool_stale_samples_by_model_;
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
