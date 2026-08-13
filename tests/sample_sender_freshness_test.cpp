#include "sample/sample_sender.h"

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct MazeServiceUpdateTestAccess {
    static void MarkSenderTransient(SampleSender& sender,
                                    const std::string& error,
                                    int retry_after_ms) {
        sender.MarkTransient(error, retry_after_ms);
    }

    static void MarkSenderHealthy(SampleSender& sender) {
        sender.MarkHealthy();
    }
};

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << message << std::endl;
        std::exit(1);
    }
}

void FillContract(const ContractConfig& config,
                  common::ContractIdentity* contract) {
    contract->set_package_name(config.package_name);
    contract->set_package_version(config.package_version);
    contract->mutable_source_digest()->set_algorithm(
        common::DIGEST_ALGORITHM_SHA256);
    contract->mutable_source_digest()->set_hex(config.source_digest.hex);
    contract->mutable_artifact_digest()->set_algorithm(
        common::DIGEST_ALGORITHM_SHA256);
    contract->mutable_artifact_digest()->set_hex(config.artifact_digest.hex);
    contract->set_platform(config.platform);
    contract->set_generator_identity(config.generator_identity);
}

class FreshnessRejectingDistributor final
    : public training::SampleDistributorService::Service {
public:
    explicit FreshnessRejectingDistributor(ContractConfig contract)
        : contract_(std::move(contract)) {}

    grpc::Status GetStatus(
        grpc::ServerContext*, const training::DistributorStatusReq*,
        training::DistributorStatusRsp* response) override {
        FillContract(contract_, response->mutable_contract());
        response->mutable_distributor()->set_component("sample-distributor");
        response->mutable_distributor()->set_instance_id("test-distributor");
        response->mutable_distributor()->set_lifecycle_epoch(1);
        response->set_ready(true);
        response->set_ingress_ready(true);
        return grpc::Status::OK;
    }

    grpc::Status AcquireSampleCredit(
        grpc::ServerContext*, const training::AcquireSampleCreditReq* request,
        training::SampleCreditGrant* response) override {
        const int call = acquire_calls_.fetch_add(1);
        response->set_request_id(request->request_id());
        response->set_batch_id(request->batch_id());
        response->mutable_distributor()->set_component("sample-distributor");
        response->mutable_distributor()->set_instance_id("test-distributor");
        response->mutable_distributor()->set_lifecycle_epoch(1);
        if (call == 0) {
            response->set_ret_code(1);
            response->set_result(
                training::SAMPLE_CREDIT_RESULT_WAIT_INFLIGHT_LIMIT);
            response->set_retry_after_ms(5);
            response->set_message("wait for current demand window");
        } else {
            response->set_ret_code(-1);
            response->set_result(
                training::SAMPLE_CREDIT_RESULT_REJECTED_FRESHNESS);
            response->set_message(
                "sample is outside the demand freshness window");
        }
        return grpc::Status::OK;
    }

private:
    ContractConfig contract_;
    std::atomic<int> acquire_calls_{0};
};

class CapacityWaitingDistributor final
    : public training::SampleDistributorService::Service {
public:
    explicit CapacityWaitingDistributor(ContractConfig contract)
        : contract_(std::move(contract)) {}

    grpc::Status GetStatus(
        grpc::ServerContext*, const training::DistributorStatusReq*,
        training::DistributorStatusRsp* response) override {
        FillContract(contract_, response->mutable_contract());
        response->mutable_distributor()->set_component("sample-distributor");
        response->mutable_distributor()->set_instance_id("wait-distributor");
        response->mutable_distributor()->set_lifecycle_epoch(1);
        response->set_ready(true);
        response->set_ingress_ready(true);
        return grpc::Status::OK;
    }

    grpc::Status AcquireSampleCredit(
        grpc::ServerContext*, const training::AcquireSampleCreditReq* request,
        training::SampleCreditGrant* response) override {
        response->set_request_id(request->request_id());
        response->set_batch_id(request->batch_id());
        response->mutable_distributor()->set_component("sample-distributor");
        response->mutable_distributor()->set_instance_id("wait-distributor");
        response->mutable_distributor()->set_lifecycle_epoch(1);
        response->set_ret_code(1);
        response->set_result(
            training::SAMPLE_CREDIT_RESULT_WAIT_INFLIGHT_LIMIT);
        response->set_retry_after_ms(7);
        response->set_message("wait for current demand window");
        return grpc::Status::OK;
    }

private:
    ContractConfig contract_;
};

class PoolStaleReportingDistributor final
    : public training::SampleDistributorService::Service {
public:
    explicit PoolStaleReportingDistributor(ContractConfig contract)
        : contract_(std::move(contract)) {}

    grpc::Status GetStatus(
        grpc::ServerContext*, const training::DistributorStatusReq*,
        training::DistributorStatusRsp* response) override {
        FillContract(contract_, response->mutable_contract());
        response->mutable_distributor()->set_component("sample-distributor");
        response->mutable_distributor()->set_instance_id(
            "pool-stale-distributor");
        response->mutable_distributor()->set_lifecycle_epoch(1);
        response->set_ready(true);
        response->set_ingress_ready(true);
        const int64_t stale = status_calls_.fetch_add(1) == 0 ? 4 : 12;
        response->set_stale_sample_count(stale);
        auto* version = response->add_behavior_versions();
        version->mutable_behavior_policy()->set_model_version(139);
        version->set_stale_samples(stale);
        return grpc::Status::OK;
    }

private:
    ContractConfig contract_;
    std::atomic<int> status_calls_{0};
};

int64_t UnixNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

class CoherenceDistributor final
    : public training::SampleDistributorService::Service {
public:
    enum class Mode {
        Accepted,
        Duplicate,
        MalformedAcceptedThenCommittedDuplicate,
        CreditRetCode,
        CreditRequestId,
        CreditBatchId,
        CreditDigest,
        CreditSamples,
        CreditFragments,
        CreditBytes,
        CreditState,
        CreditExpired,
        CreditAuthority,
        WaitWithGrantPayload,
        FreshnessWithGrantPayload,
        PushRetCode,
        PushBatchId,
        PushAcceptedSamples,
        PushAcceptedUniqueSamples,
        PushDuplicateCounts,
        PushAuthority,
    };

    explicit CoherenceDistributor(ContractConfig contract)
        : contract_(std::move(contract)) {}

    void SetMode(Mode mode) {
        mode_.store(static_cast<int>(mode));
        acquire_calls_.store(0);
        push_calls_.store(0);
    }

    int acquire_calls() const { return acquire_calls_.load(); }
    int push_calls() const { return push_calls_.load(); }

    grpc::Status GetStatus(
        grpc::ServerContext*, const training::DistributorStatusReq*,
        training::DistributorStatusRsp* response) override {
        FillContract(contract_, response->mutable_contract());
        FillAuthority(response->mutable_distributor());
        response->set_ready(true);
        response->set_ingress_ready(true);
        return grpc::Status::OK;
    }

    grpc::Status AcquireSampleCredit(
        grpc::ServerContext*, const training::AcquireSampleCreditReq* request,
        training::SampleCreditGrant* response) override {
        const int call = acquire_calls_.fetch_add(1) + 1;
        const Mode mode = CurrentMode();
        response->set_ret_code(0);
        response->set_result(training::SAMPLE_CREDIT_RESULT_GRANTED);
        response->set_message("credit granted");
        response->set_request_id(request->request_id());
        response->set_credit_id(request->batch_id() + "-credit");
        response->set_demand_id("coherence-demand");
        response->set_demand_epoch(7);
        response->set_batch_id(request->batch_id());
        *response->mutable_payload_digest() = request->payload_digest();
        response->set_granted_samples(request->sample_count());
        response->set_granted_fragments(request->fragment_count());
        response->set_granted_estimated_bytes(request->estimated_bytes());
        response->set_expires_at_unix_ms(UnixNowMs() + 60000);
        response->set_pressure_state(training::PRESSURE_STATE_NORMAL);
        FillAuthority(response->mutable_distributor());
        response->set_state(training::SAMPLE_CREDIT_STATE_RESERVED);

        if (mode == Mode::MalformedAcceptedThenCommittedDuplicate && call > 1) {
            response->set_state(training::SAMPLE_CREDIT_STATE_COMMITTED);
            response->set_expires_at_unix_ms(UnixNowMs() - 1);
        }
        switch (mode) {
            case Mode::CreditRetCode:
                response->set_ret_code(1);
                break;
            case Mode::CreditRequestId:
                response->set_request_id("wrong-credit-request");
                break;
            case Mode::CreditBatchId:
                response->set_batch_id("wrong-credit-batch");
                break;
            case Mode::CreditDigest:
                response->mutable_payload_digest()->set_hex(
                    std::string(64, 'e'));
                break;
            case Mode::CreditSamples:
                response->set_granted_samples(request->sample_count() + 1);
                break;
            case Mode::CreditFragments:
                response->set_granted_fragments(
                    request->fragment_count() + 1);
                break;
            case Mode::CreditBytes:
                response->set_granted_estimated_bytes(
                    request->estimated_bytes() + 1);
                break;
            case Mode::CreditState:
                response->set_state(training::SAMPLE_CREDIT_STATE_EXPIRED);
                break;
            case Mode::CreditExpired:
                response->set_expires_at_unix_ms(UnixNowMs() - 1);
                break;
            case Mode::CreditAuthority:
                response->mutable_distributor()->set_instance_id(
                    "changed-distributor");
                break;
            case Mode::WaitWithGrantPayload:
                response->set_ret_code(1);
                response->set_result(
                    training::SAMPLE_CREDIT_RESULT_WAIT_CAPACITY);
                response->set_retry_after_ms(5);
                break;
            case Mode::FreshnessWithGrantPayload:
                response->set_ret_code(-1);
                response->set_result(
                    training::SAMPLE_CREDIT_RESULT_REJECTED_FRESHNESS);
                break;
            default:
                break;
        }
        return grpc::Status::OK;
    }

    grpc::Status PushSamples(
        grpc::ServerContext*, const training::PushSamplesReq* request,
        training::PushSamplesRsp* response) override {
        const int call = push_calls_.fetch_add(1) + 1;
        const Mode mode = CurrentMode();
        response->set_ret_code(0);
        response->set_result(training::PUSH_RESULT_ACCEPTED);
        response->set_message("accepted");
        response->set_batch_id(request->batch().batch_id());
        response->set_accepted_samples(request->batch().samples_size());
        response->set_accepted_unique_samples(
            request->batch().samples_size());
        response->set_pressure_state(training::PRESSURE_STATE_NORMAL);
        FillAuthority(response->mutable_distributor());

        if (mode == Mode::Duplicate ||
            (mode == Mode::MalformedAcceptedThenCommittedDuplicate &&
             call > 1)) {
            response->set_result(training::PUSH_RESULT_DUPLICATE);
            response->set_message("batch already accepted");
            response->set_accepted_samples(0);
            response->set_accepted_unique_samples(0);
        }
        switch (mode) {
            case Mode::MalformedAcceptedThenCommittedDuplicate:
                if (call == 1) response->set_accepted_samples(0);
                break;
            case Mode::PushRetCode:
                response->set_ret_code(-1);
                break;
            case Mode::PushBatchId:
                response->set_batch_id("wrong-push-batch");
                break;
            case Mode::PushAcceptedSamples:
                response->set_accepted_samples(
                    request->batch().samples_size() - 1);
                break;
            case Mode::PushAcceptedUniqueSamples:
                response->set_accepted_unique_samples(0);
                break;
            case Mode::PushDuplicateCounts:
                response->set_result(training::PUSH_RESULT_DUPLICATE);
                response->set_accepted_samples(
                    request->batch().samples_size());
                response->set_accepted_unique_samples(
                    request->batch().samples_size());
                break;
            case Mode::PushAuthority:
                response->mutable_distributor()->set_lifecycle_epoch(8);
                break;
            default:
                break;
        }
        return grpc::Status::OK;
    }

private:
    static void FillAuthority(common::ServiceInstanceIdentity* identity) {
        identity->set_component("sample-distributor");
        identity->set_instance_id("coherence-distributor");
        identity->set_lifecycle_epoch(7);
    }

    Mode CurrentMode() const {
        return static_cast<Mode>(mode_.load());
    }

    ContractConfig contract_;
    std::atomic<int> mode_{static_cast<int>(Mode::Accepted)};
    std::atomic<int> acquire_calls_{0};
    std::atomic<int> push_calls_{0};
};

class TransientRecoveryDistributor final
    : public training::SampleDistributorService::Service {
public:
    enum class Mode {
        StatusFailure,
        AcquireFailure,
        PushFailureAfterApply,
    };

    explicit TransientRecoveryDistributor(ContractConfig contract)
        : contract_(std::move(contract)) {}

    void SetMode(Mode mode, grpc::StatusCode transport_code) {
        mode_.store(static_cast<int>(mode));
        transport_code_.store(static_cast<int>(transport_code));
        recovery_allowed_.store(false);
        status_calls_.store(0);
        acquire_calls_.store(0);
        push_calls_.store(0);
        push_committed_.store(false);
        identity_mismatch_.store(false);
        std::lock_guard<std::mutex> lock(identity_mutex_);
        request_id_.clear();
        batch_id_.clear();
        digest_.clear();
    }

    void AllowRecovery() { recovery_allowed_.store(true); }
    int status_calls() const { return status_calls_.load(); }
    int acquire_calls() const { return acquire_calls_.load(); }
    int push_calls() const { return push_calls_.load(); }
    bool identity_mismatch() const { return identity_mismatch_.load(); }

    grpc::Status GetStatus(
        grpc::ServerContext*, const training::DistributorStatusReq*,
        training::DistributorStatusRsp* response) override {
        const int call = status_calls_.fetch_add(1) + 1;
        if (CurrentMode() == Mode::StatusFailure && call > 1 &&
            !recovery_allowed_.load()) {
            return grpc::Status(
                CurrentTransportCode(),
                "injected GetStatus transport outage");
        }
        FillStatus(response);
        return grpc::Status::OK;
    }

    grpc::Status AcquireSampleCredit(
        grpc::ServerContext*, const training::AcquireSampleCreditReq* request,
        training::SampleCreditGrant* response) override {
        acquire_calls_.fetch_add(1);
        TrackIdentity(request->request_id(), request->batch_id(),
                      request->payload_digest().hex());
        if ((CurrentMode() == Mode::AcquireFailure ||
             (CurrentMode() == Mode::PushFailureAfterApply &&
              push_committed_.load())) &&
            !recovery_allowed_.load()) {
            return grpc::Status(
                CurrentTransportCode(),
                "injected AcquireSampleCredit transport failure");
        }

        response->set_ret_code(0);
        response->set_result(training::SAMPLE_CREDIT_RESULT_GRANTED);
        response->set_message("credit granted");
        response->set_request_id(request->request_id());
        response->set_credit_id(request->batch_id() + "-credit");
        response->set_demand_id("transient-demand");
        response->set_demand_epoch(9);
        response->set_batch_id(request->batch_id());
        *response->mutable_payload_digest() = request->payload_digest();
        response->set_granted_samples(request->sample_count());
        response->set_granted_fragments(request->fragment_count());
        response->set_granted_estimated_bytes(request->estimated_bytes());
        response->set_expires_at_unix_ms(UnixNowMs() + 60000);
        response->set_pressure_state(training::PRESSURE_STATE_NORMAL);
        FillAuthority(response->mutable_distributor());
        response->set_state(
            push_committed_.load()
                ? training::SAMPLE_CREDIT_STATE_COMMITTED
                : training::SAMPLE_CREDIT_STATE_RESERVED);
        return grpc::Status::OK;
    }

    grpc::Status PushSamples(
        grpc::ServerContext*, const training::PushSamplesReq* request,
        training::PushSamplesRsp* response) override {
        push_calls_.fetch_add(1);
        TrackIdentity(request->batch().batch_id() + "-credit",
                      request->batch().batch_id(),
                      request->batch().payload_digest().hex());
        if (CurrentMode() == Mode::PushFailureAfterApply &&
            !push_committed_.exchange(true)) {
            return grpc::Status(
                CurrentTransportCode(),
                "injected PushSamples response loss after apply");
        }
        response->set_ret_code(0);
        response->set_result(
            push_committed_.load()
                ? training::PUSH_RESULT_DUPLICATE
                : training::PUSH_RESULT_ACCEPTED);
        response->set_message(
            push_committed_.load() ? "batch already accepted" : "accepted");
        response->set_batch_id(request->batch().batch_id());
        response->set_accepted_samples(
            push_committed_.load() ? 0 : request->batch().samples_size());
        response->set_accepted_unique_samples(
            push_committed_.load() ? 0 : request->batch().samples_size());
        response->set_pressure_state(training::PRESSURE_STATE_NORMAL);
        FillAuthority(response->mutable_distributor());
        return grpc::Status::OK;
    }

private:
    Mode CurrentMode() const {
        return static_cast<Mode>(mode_.load());
    }

    grpc::StatusCode CurrentTransportCode() const {
        return static_cast<grpc::StatusCode>(transport_code_.load());
    }

    void FillStatus(training::DistributorStatusRsp* response) const {
        FillContract(contract_, response->mutable_contract());
        FillAuthority(response->mutable_distributor());
        response->set_ready(true);
        response->set_ingress_ready(true);
    }

    static void FillAuthority(common::ServiceInstanceIdentity* identity) {
        identity->set_component("sample-distributor");
        identity->set_instance_id("transient-distributor");
        identity->set_lifecycle_epoch(9);
    }

    void TrackIdentity(const std::string& request_id,
                       const std::string& batch_id,
                       const std::string& digest) {
        std::lock_guard<std::mutex> lock(identity_mutex_);
        if (request_id_.empty()) {
            request_id_ = request_id;
            batch_id_ = batch_id;
            digest_ = digest;
            return;
        }
        if (request_id_ != request_id || batch_id_ != batch_id ||
            digest_ != digest) {
            identity_mismatch_.store(true);
        }
    }

    ContractConfig contract_;
    std::atomic<int> mode_{static_cast<int>(Mode::StatusFailure)};
    std::atomic<int> transport_code_{
        static_cast<int>(grpc::StatusCode::UNAVAILABLE)};
    std::atomic<bool> recovery_allowed_{false};
    std::atomic<bool> push_committed_{false};
    std::atomic<bool> identity_mismatch_{false};
    std::atomic<int> status_calls_{0};
    std::atomic<int> acquire_calls_{0};
    std::atomic<int> push_calls_{0};
    std::mutex identity_mutex_;
    std::string request_id_;
    std::string batch_id_;
    std::string digest_;
};

training::SampleBatch CoherenceBatch(const std::string& batch_id) {
    training::SampleBatch batch;
    batch.set_batch_id(batch_id);
    batch.mutable_payload_digest()->set_algorithm(
        common::DIGEST_ALGORITHM_SHA256);
    batch.mutable_payload_digest()->set_hex(std::string(64, 'd'));
    batch.set_created_at_unix_ms(UnixNowMs());
    batch.mutable_behavior_policy()->set_model_version(1);
    batch.add_samples();
    batch.add_samples();
    return batch;
}

void VerifyFrameAtomicCapacityWait(AIServerConfig config) {
    CapacityWaitingDistributor service(config.contract);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "capacity distributor did not start");

    config.sample_output.port = port;
    config.sample_output.outbound_max_fragments = 5;
    config.sample_output.outbound_max_estimated_bytes = 5 * 1024 * 1024;
    SampleSender sender(config);
    Require(sender.Start(), "capacity sender did not start");

    training::SampleBatch batch;
    batch.set_batch_id("capacity-batch");
    batch.mutable_behavior_policy()->set_model_version(1);
    batch.add_samples();
    Require(sender.Enqueue(batch), "capacity batch was not enqueued");

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(2);
    while (!sender.IsWaitingForTrainingCapacity() &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const auto snapshot = sender.GetSnapshot();
    Require(sender.IsWaitingForTrainingCapacity(),
            "one full Agent frame was not reserved before mutation");
    Require(snapshot.training_capacity_wait && !snapshot.degraded,
            "retryable capacity wait degraded the sender");
    Require(snapshot.delivery_state ==
                SampleSender::DeliveryState::kFlowWait,
            "capacity wait did not enter the flow-control state");
    Require(sender.TrainingCapacityRetryAfterMs() == 7,
            "downstream retry interval was not preserved");

    sender.MarkDegraded("test shutdown");
    sender.StopAndDrain();
    server->Shutdown();
    server->Wait();
}

void VerifyBatchSetReservationAtomic(AIServerConfig config) {
    CapacityWaitingDistributor service(config.contract);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "batch-set distributor did not start");

    config.sample_output.port = port;
    config.sample_output.outbound_max_fragments = 8;
    config.sample_output.outbound_max_estimated_bytes = 8 * 1024 * 1024;
    config.sample_output.drain_timeout_ms = 20;
    SampleSender sender(config);
    Require(sender.Start(), "batch-set sender did not start");

    std::vector<training::SampleBatch> batches;
    for (int index = 0; index < 2; ++index) {
        training::SampleBatch batch;
        batch.set_batch_id("atomic-batch-" + std::to_string(index));
        batch.mutable_behavior_policy()->set_model_version(1);
        batch.add_samples();
        batches.push_back(std::move(batch));
    }
    uint64_t reservation = 0;
    std::string error;
    Require(sender.ReserveEnqueueBatchSet(
                batches, reservation, error) ==
                SampleSender::ReservationResult::kReserved,
            "complete batch set reserves: " + error);
    Require(sender.HasEnqueueReservation(reservation),
            "reservation identity is queryable before commit");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    auto snapshot = sender.GetSnapshot();
    Require(snapshot.queue_fragments == 0 && snapshot.queue_samples == 0,
            "reserved batch set is invisible to the sender");

    sender.CancelEnqueueBatchSet(reservation);
    Require(!sender.HasEnqueueReservation(reservation),
            "cancellation releases the whole reservation");
    snapshot = sender.GetSnapshot();
    Require(snapshot.queue_fragments == 0 && snapshot.queue_samples == 0,
            "cancellation exposes no partial queue prefix");

    Require(sender.ReserveEnqueueBatchSet(
                batches, reservation, error) ==
                SampleSender::ReservationResult::kReserved,
            "batch set reserves again after cancellation: " + error);
    MazeServiceUpdateTestAccess::MarkSenderTransient(
        sender, "injected state change between reserve and commit", 19);
    Require(!sender.HasEnqueueReservation(reservation),
            "delivery generation change invalidates the live token");
    Require(sender.CommitEnqueueBatchSet(reservation, error) ==
                SampleSender::CommitResult::kRetryableUnavailable,
            "transient commit gate returns retryable: " + error);
    snapshot = sender.GetSnapshot();
    Require(snapshot.queue_fragments == 0 && snapshot.queue_samples == 0 &&
                snapshot.transient_retry && !snapshot.degraded &&
                !sender.HasEnqueueReservation(reservation),
            "reserve/commit race exposed a partial batch set");

    MazeServiceUpdateTestAccess::MarkSenderHealthy(sender);
    Require(sender.ReserveEnqueueBatchSet(
                batches, reservation, error) ==
                SampleSender::ReservationResult::kReserved,
            "batch set reserves after delivery recovery: " + error);
    Require(sender.SealEnqueueBatchSet(reservation, error) ==
                SampleSender::SealResult::kSealed,
            "batch set seals before the point of no return: " + error);
    MazeServiceUpdateTestAccess::MarkSenderTransient(
        sender, "injected state change after reservation seal", 23);
    Require(sender.CommitEnqueueBatchSet(reservation, error) ==
                SampleSender::CommitResult::kCommitted,
            "sealed batch set rolls forward across delivery change: " + error);
    snapshot = sender.GetSnapshot();
    Require(snapshot.queue_fragments == 2 && snapshot.queue_samples == 2 &&
                snapshot.transient_retry && !snapshot.degraded,
            "one splice publishes the complete sealed batch set while the "
            "transport recovers");

    sender.MarkDegraded("test shutdown");
    sender.StopAndDrain();
    server->Shutdown();
    server->Wait();
}

void VerifyPoolStaleStatusPolling(AIServerConfig config) {
    PoolStaleReportingDistributor service(config.contract);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "pool stale distributor did not start");

    config.sample_output.port = port;
    config.sample_output.status_poll_interval_ms = 10;
    SampleSender sender(config);
    Require(sender.Start(), "pool stale sender did not start");

    SampleSender::Snapshot snapshot;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(2);
    do {
        snapshot = sender.GetSnapshot();
        if (snapshot.pool_stale_count == 8) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } while (std::chrono::steady_clock::now() < deadline);

    Require(snapshot.pool_stale_count == 8,
            "pool stale status did not exclude the startup baseline");
    Require(snapshot.pool_stale_samples_by_model.at(139) == 8,
            "pool stale status lost behavior model identity");
    Require(snapshot.ready && !snapshot.degraded &&
                snapshot.last_error.empty(),
            "valid pool stale status degraded the sender");

    Require(sender.StopAndDrain(), "pool stale sender did not drain");
    server->Shutdown();
    server->Wait();
}

SampleSender::Snapshot WaitForSender(
    SampleSender& sender,
    bool expect_commit) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(2);
    SampleSender::Snapshot snapshot;
    do {
        snapshot = sender.GetSnapshot();
        if ((expect_commit && snapshot.queue_fragments == 0) ||
            (!expect_commit && snapshot.degraded)) {
            return snapshot;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } while (std::chrono::steady_clock::now() < deadline);
    return sender.GetSnapshot();
}

SampleSender::Snapshot WaitForDeliveryState(
    SampleSender& sender,
    SampleSender::DeliveryState expected) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(2);
    SampleSender::Snapshot snapshot;
    do {
        snapshot = sender.GetSnapshot();
        if (snapshot.delivery_state == expected) return snapshot;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    } while (std::chrono::steady_clock::now() < deadline);
    return sender.GetSnapshot();
}

void VerifyTransientTransportRecovery(AIServerConfig config) {
    TransientRecoveryDistributor service(config.contract);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "transient recovery distributor did not start");

    config.sample_output.port = port;
    config.sample_output.max_attempts = 1;
    config.sample_output.rpc_timeout_ms = 100;
    config.sample_output.drain_timeout_ms = 1000;

    const std::vector<grpc::StatusCode> retryable_codes = {
        grpc::StatusCode::CANCELLED,
        grpc::StatusCode::UNKNOWN,
        grpc::StatusCode::DEADLINE_EXCEEDED,
        grpc::StatusCode::RESOURCE_EXHAUSTED,
        grpc::StatusCode::ABORTED,
        grpc::StatusCode::INTERNAL,
        grpc::StatusCode::UNAVAILABLE,
    };
    for (const auto code : retryable_codes) {
        const std::string suffix =
            std::to_string(static_cast<int>(code));

        service.SetMode(
            TransientRecoveryDistributor::Mode::StatusFailure, code);
        config.sample_output.status_poll_interval_ms = 5;
        SampleSender status_sender(config);
        Require(status_sender.Start(),
                "status recovery sender did not start: " + suffix);
        auto snapshot = WaitForDeliveryState(
            status_sender, SampleSender::DeliveryState::kTransientRetry);
        Require(snapshot.transient_retry && !snapshot.degraded &&
                    snapshot.ready && snapshot.queue_fragments == 0 &&
                    service.status_calls() >= 2,
                "GetStatus outcome-unknown code became terminal: " + suffix);
        service.AllowRecovery();
        snapshot = WaitForDeliveryState(
            status_sender, SampleSender::DeliveryState::kHealthy);
        Require(snapshot.delivery_state ==
                        SampleSender::DeliveryState::kHealthy &&
                    snapshot.ready && !snapshot.degraded &&
                    snapshot.last_error.empty(),
                "GetStatus recovery did not restore the sender: " + suffix);
        Require(status_sender.StopAndDrain(),
                "status recovery sender did not stop: " + suffix);

        service.SetMode(
            TransientRecoveryDistributor::Mode::AcquireFailure, code);
        config.sample_output.status_poll_interval_ms = 10000;
        SampleSender acquire_sender(config);
        Require(acquire_sender.Start(),
                "acquire recovery sender did not start: " + suffix);
        Require(acquire_sender.Enqueue(
                    CoherenceBatch("transient-acquire-" + suffix)),
                "acquire recovery batch did not enqueue: " + suffix);
        snapshot = WaitForDeliveryState(
            acquire_sender, SampleSender::DeliveryState::kTransientRetry);
        Require(snapshot.transient_retry && !snapshot.degraded &&
                    snapshot.queue_fragments == 1 &&
                    snapshot.queue_samples == 2 &&
                    snapshot.accepted_unique_samples == 0,
                "Acquire outcome-unknown code lost the batch: " + suffix);
        service.AllowRecovery();
        snapshot = WaitForSender(acquire_sender, true);
        Require(snapshot.queue_fragments == 0 && !snapshot.degraded &&
                    snapshot.delivery_state ==
                        SampleSender::DeliveryState::kHealthy &&
                    snapshot.accepted_unique_samples == 2 &&
                    service.acquire_calls() >= 2 &&
                    !service.identity_mismatch(),
                "Acquire recovery changed exact identity: " + suffix);
        Require(acquire_sender.StopAndDrain(),
                "acquire recovery sender did not drain: " + suffix);

        service.SetMode(
            TransientRecoveryDistributor::Mode::PushFailureAfterApply, code);
        SampleSender push_sender(config);
        Require(push_sender.Start(),
                "push recovery sender did not start: " + suffix);
        Require(push_sender.Enqueue(
                    CoherenceBatch("transient-push-" + suffix)),
                "push recovery batch did not enqueue: " + suffix);
        snapshot = WaitForDeliveryState(
            push_sender, SampleSender::DeliveryState::kTransientRetry);
        Require(snapshot.transient_retry && !snapshot.degraded &&
                    snapshot.queue_fragments == 1 &&
                    snapshot.queue_samples == 2 &&
                    snapshot.accepted_unique_samples == 0 &&
                    service.push_calls() == 1,
                "Push outcome-unknown code disposed the front: " + suffix);
        service.AllowRecovery();
        snapshot = WaitForSender(push_sender, true);
        Require(snapshot.queue_fragments == 0 && !snapshot.degraded &&
                    snapshot.delivery_state ==
                        SampleSender::DeliveryState::kHealthy &&
                    snapshot.accepted_unique_samples == 2 &&
                    snapshot.accepted_unique_batches == 1 &&
                    snapshot.duplicate_push_attempt_count == 1 &&
                    service.acquire_calls() >= 2 &&
                    service.push_calls() == 2 &&
                    !service.identity_mismatch(),
                "Push outcome-unknown recovery did not converge through "
                "COMMITTED/DUPLICATE: " + suffix);
        Require(push_sender.StopAndDrain(),
                "push recovery sender did not drain: " + suffix);
    }

    service.SetMode(
        TransientRecoveryDistributor::Mode::PushFailureAfterApply,
        grpc::StatusCode::DATA_LOSS);
    SampleSender rewritten_status_sender(config);
    Require(rewritten_status_sender.Start(),
            "DATA_LOSS response-rewrite sender did not start");
    Require(rewritten_status_sender.Enqueue(
                CoherenceBatch("push-data-loss-after-apply")),
            "DATA_LOSS response-rewrite batch did not enqueue");
    auto rewritten_snapshot = WaitForDeliveryState(
        rewritten_status_sender,
        SampleSender::DeliveryState::kTransientRetry);
    Require(rewritten_snapshot.queue_fragments == 1 &&
                rewritten_snapshot.queue_samples == 2 &&
                rewritten_snapshot.accepted_unique_samples == 0 &&
                !rewritten_snapshot.degraded && service.push_calls() == 1,
            "non-whitelisted Push status disposed an outcome-unknown batch");
    service.AllowRecovery();
    rewritten_snapshot = WaitForSender(rewritten_status_sender, true);
    Require(rewritten_snapshot.queue_fragments == 0 &&
                rewritten_snapshot.queue_samples == 0 &&
                rewritten_snapshot.accepted_unique_samples == 2 &&
                rewritten_snapshot.accepted_unique_batches == 1 &&
                rewritten_snapshot.duplicate_push_attempt_count == 1 &&
                rewritten_snapshot.final_drop_unique_samples == 0 &&
                rewritten_snapshot.final_drop_unique_batches == 0 &&
                rewritten_snapshot.unresolved_push_outcome_unknown_samples == 0 &&
                rewritten_snapshot.unresolved_push_outcome_unknown_batches == 0 &&
                service.push_calls() == 2 &&
                !service.identity_mismatch(),
            "DATA_LOSS-after-apply did not converge through the exact "
            "COMMITTED/DUPLICATE identity");
    Require(2 == rewritten_snapshot.accepted_unique_samples +
                     rewritten_snapshot.queue_samples +
                     rewritten_snapshot.final_drop_unique_samples &&
                rewritten_snapshot.unresolved_push_outcome_unknown_samples == 0,
            "DATA_LOSS-after-apply violates produced = accepted + outbound + "
            "known final-drop conservation");
    Require(rewritten_status_sender.StopAndDrain(),
            "DATA_LOSS response-rewrite sender did not drain");

    service.SetMode(
        TransientRecoveryDistributor::Mode::PushFailureAfterApply,
        grpc::StatusCode::UNAVAILABLE);
    auto unresolved_config = config;
    unresolved_config.sample_output.drain_timeout_ms = 10;
    SampleSender unresolved_sender(unresolved_config);
    Require(unresolved_sender.Start(),
            "unresolved Push outcome sender did not start");
    Require(unresolved_sender.Enqueue(
                CoherenceBatch("push-unresolved-at-drain")),
            "unresolved Push outcome batch did not enqueue");
    auto unresolved_snapshot = WaitForDeliveryState(
        unresolved_sender, SampleSender::DeliveryState::kTransientRetry);
    Require(unresolved_snapshot.queue_fragments == 1 &&
                unresolved_snapshot.queue_samples == 2,
            "unresolved Push outcome did not retain the queue front");
    Require(!unresolved_sender.StopAndDrain(),
            "unresolved Push outcome unexpectedly reported a clean drain");
    unresolved_snapshot = unresolved_sender.GetSnapshot();
    Require(unresolved_snapshot.terminal_fault &&
                unresolved_snapshot.queue_fragments == 0 &&
                unresolved_snapshot.final_drop_unique_samples == 0 &&
                unresolved_snapshot.final_drop_unique_batches == 0 &&
                unresolved_snapshot.unresolved_push_outcome_unknown_samples ==
                    2 &&
                unresolved_snapshot.unresolved_push_outcome_unknown_batches ==
                    1,
            "drain deadline misclassified an outcome-unknown remote commit as "
            "a known final drop");

    server->Shutdown();
    server->Wait();
}

void VerifySampleResponseCoherence(AIServerConfig config) {
    CoherenceDistributor service(config.contract);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "coherence distributor did not start");

    config.sample_output.port = port;
    config.sample_output.max_attempts = 1;
    config.sample_output.drain_timeout_ms = 20;
    config.sample_output.status_poll_interval_ms = 10000;

    const auto verify_success = [&](CoherenceDistributor::Mode mode,
                                    const std::string& label,
                                    bool duplicate) {
        service.SetMode(mode);
        SampleSender sender(config);
        Require(sender.Start(), label + ": sender did not start");
        Require(sender.Enqueue(CoherenceBatch(label)),
                label + ": batch did not enqueue");
        const auto snapshot = WaitForSender(sender, true);
        Require(snapshot.queue_fragments == 0 && snapshot.queue_samples == 0 &&
                    snapshot.ready && !snapshot.degraded &&
                    snapshot.last_error.empty(),
                label + ": coherent response did not commit cleanly");
        Require(service.acquire_calls() == 1 && service.push_calls() == 1,
                label + ": coherent response used an unexpected RPC count");
        Require(snapshot.credit_grant_count == 1 &&
                    snapshot.push_attempt_count == 1 &&
                    snapshot.push_rpc_count == 1,
                label + ": coherent response counters are unbalanced");
        Require(snapshot.accepted_unique_samples == 2 &&
                    snapshot.accepted_unique_batches == 1 &&
                    snapshot.duplicate_push_attempt_count ==
                        (duplicate ? 1 : 0),
                label +
                    ": validated remote acceptance lost unique sample counts");
        Require(2 == snapshot.accepted_unique_samples +
                         snapshot.queue_samples +
                         snapshot.final_drop_unique_samples,
                label + ": producer disposition does not conserve samples");
        Require(sender.StopAndDrain(), label + ": sender did not drain");
    };
    verify_success(CoherenceDistributor::Mode::Accepted,
                   "coherent-accepted", false);
    verify_success(CoherenceDistributor::Mode::Duplicate,
                   "coherent-duplicate", true);

    service.SetMode(
        CoherenceDistributor::Mode::MalformedAcceptedThenCommittedDuplicate);
    auto retry_config = config;
    retry_config.sample_output.max_attempts = 2;
    SampleSender retry_sender(retry_config);
    Require(retry_sender.Start(), "coherence retry sender did not start");
    Require(retry_sender.Enqueue(CoherenceBatch("coherence-retry")),
            "coherence retry batch did not enqueue");
    const auto retry_snapshot = WaitForSender(retry_sender, true);
    Require(retry_snapshot.queue_fragments == 0 &&
                retry_snapshot.queue_samples == 0 &&
                retry_snapshot.accepted_unique_samples == 2 &&
                retry_snapshot.accepted_unique_batches == 1 &&
                retry_snapshot.duplicate_push_attempt_count == 1 &&
                retry_snapshot.retry_attempt_count >= 1 &&
                retry_snapshot.credit_reacquire_count >= 1 &&
                retry_snapshot.credit_grant_count == 2 &&
                retry_snapshot.push_attempt_count == 2 &&
                retry_snapshot.push_rpc_count == 2 &&
                !retry_snapshot.degraded &&
                service.acquire_calls() == 2 && service.push_calls() == 2,
            "malformed positive response did not converge through exact "
            "COMMITTED/DUPLICATE retry");
    Require(retry_sender.StopAndDrain(),
            "coherence retry sender did not drain");

    struct FailureCase {
        CoherenceDistributor::Mode mode;
        const char* label;
        bool reaches_push;
    };
    const std::vector<FailureCase> failures = {
        {CoherenceDistributor::Mode::CreditRetCode,
         "credit-ret-code", false},
        {CoherenceDistributor::Mode::CreditRequestId,
         "credit-request-id", false},
        {CoherenceDistributor::Mode::CreditBatchId,
         "credit-batch-id", false},
        {CoherenceDistributor::Mode::CreditDigest,
         "credit-payload-digest", false},
        {CoherenceDistributor::Mode::CreditSamples,
         "credit-sample-count", false},
        {CoherenceDistributor::Mode::CreditFragments,
         "credit-fragment-count", false},
        {CoherenceDistributor::Mode::CreditBytes,
         "credit-estimated-bytes", false},
        {CoherenceDistributor::Mode::CreditState,
         "credit-state", false},
        {CoherenceDistributor::Mode::CreditExpired,
         "credit-expiry", false},
        {CoherenceDistributor::Mode::CreditAuthority,
         "credit-authority", false},
        {CoherenceDistributor::Mode::WaitWithGrantPayload,
         "credit-wait-with-grant-payload", false},
        {CoherenceDistributor::Mode::FreshnessWithGrantPayload,
         "credit-freshness-with-grant-payload", false},
        {CoherenceDistributor::Mode::PushRetCode,
         "push-ret-code", true},
        {CoherenceDistributor::Mode::PushBatchId,
         "push-batch-id", true},
        {CoherenceDistributor::Mode::PushAcceptedSamples,
         "push-accepted-samples", true},
        {CoherenceDistributor::Mode::PushAcceptedUniqueSamples,
         "push-accepted-unique-samples", true},
        {CoherenceDistributor::Mode::PushDuplicateCounts,
         "push-duplicate-counts", true},
        {CoherenceDistributor::Mode::PushAuthority,
         "push-authority", true},
    };
    for (const auto& failure : failures) {
        service.SetMode(failure.mode);
        SampleSender sender(config);
        const std::string label = failure.label;
        Require(sender.Start(), label + ": sender did not start");
        Require(sender.Enqueue(CoherenceBatch(label)),
                label + ": batch did not enqueue");
        if (failure.reaches_push) {
            auto snapshot = WaitForDeliveryState(
                sender, SampleSender::DeliveryState::kTransientRetry);
            Require(snapshot.transient_retry && !snapshot.degraded &&
                        snapshot.queue_fragments == 1 &&
                        snapshot.queue_samples == 2 &&
                        snapshot.accepted_unique_samples == 0 &&
                        snapshot.accepted_unique_batches == 0 &&
                        snapshot.duplicate_push_attempt_count == 0 &&
                        snapshot.push_rpc_count >= 1,
                    label +
                        ": malformed Push reply was treated as conclusive");
            Require(!sender.StopAndDrain(),
                    label + ": unresolved malformed Push unexpectedly drained");
            snapshot = sender.GetSnapshot();
            Require(snapshot.unresolved_push_outcome_unknown_samples == 2 &&
                        snapshot.unresolved_push_outcome_unknown_batches == 1 &&
                        snapshot.final_drop_unique_samples == 0 &&
                        snapshot.final_drop_unique_batches == 0,
                    label +
                        ": unresolved malformed Push was misclassified as a "
                        "known final drop");
        } else {
            const auto snapshot = WaitForSender(sender, false);
            Require(snapshot.degraded && !snapshot.last_error.empty() &&
                        snapshot.terminal_fault &&
                        snapshot.delivery_state ==
                            SampleSender::DeliveryState::kTerminalFault &&
                        snapshot.queue_fragments == 1 &&
                        snapshot.queue_samples == 2 &&
                        snapshot.accepted_unique_samples == 0 &&
                        snapshot.accepted_unique_batches == 0 &&
                        snapshot.duplicate_push_attempt_count == 0 &&
                        snapshot.producer_stale_count == 0 &&
                        snapshot.credit_grant_count == 0 &&
                        snapshot.push_attempt_count == 0 &&
                        snapshot.push_rpc_count == 0,
                    label +
                        ": malformed credit response lost or miscounted batch");
            Require(service.acquire_calls() == 1 &&
                        service.push_calls() == 0,
                    label +
                        ": credit validation crossed the Push RPC boundary");
            sender.StopAndDrain();
        }
    }

    server->Shutdown();
    server->Wait();
}

}  // namespace

int main() {
    AIServerConfig config;
    config.contract.source_digest.hex = std::string(64, 'a');
    config.contract.artifact_digest.hex = std::string(64, 'b');
    config.contract.generator_identity = std::string(64, 'c');
    config.sample_output.host = "127.0.0.1";
    config.sample_output.health_timeout_ms = 1000;
    config.sample_output.rpc_timeout_ms = 1000;
    config.sample_output.max_attempts = 1;
    config.sample_output.drain_timeout_ms = 1000;

    FreshnessRejectingDistributor service(config.contract);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0, "test distributor did not start");
    config.sample_output.port = port;

    SampleSender sender(config);
    Require(sender.Start(), "sample sender did not start");

    training::SampleBatch batch;
    batch.set_batch_id("stale-batch");
    constexpr uint64_t kHighBehaviorVersion =
        std::numeric_limits<uint64_t>::max();
    batch.mutable_behavior_policy()->set_model_version(kHighBehaviorVersion);
    for (int index = 0; index < 8; ++index) batch.add_samples();
    Require(sender.Enqueue(batch), "stale batch was not enqueued");

    SampleSender::Snapshot snapshot;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(2);
    do {
        snapshot = sender.GetSnapshot();
        if (snapshot.producer_stale_count == 8) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);

    Require(snapshot.producer_stale_count == 8,
            "producer stale disposition did not count samples");
    Require(snapshot.producer_stale_samples_by_model.at(
                kHighBehaviorVersion) == 8,
            "producer stale disposition lost behavior model identity");
    Require(snapshot.queue_samples == 0 && snapshot.queue_fragments == 0,
            "producer stale fragment remained queued");
    Require(snapshot.ready && !snapshot.degraded,
            "producer stale disposition degraded the sender");
    Require(!snapshot.training_capacity_wait,
            "producer stale disposition left capacity WAIT active");
    Require(snapshot.last_error.empty(),
            "producer stale disposition leaked a service error");

    Require(sender.StopAndDrain(), "sample sender did not drain");
    server->Shutdown();
    server->Wait();
    VerifyFrameAtomicCapacityWait(config);
    VerifyBatchSetReservationAtomic(config);
    VerifyPoolStaleStatusPolling(config);
    VerifySampleResponseCoherence(config);
    VerifyTransientTransportRecovery(config);
    return 0;
}
