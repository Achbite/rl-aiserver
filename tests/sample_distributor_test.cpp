#include "sample/sample_sender.h"

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

int64_t UnixNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
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
    contract->mutable_artifact_digest()->set_hex(
        config.artifact_digest.hex);
    contract->set_platform(config.platform);
    contract->set_generator_identity(config.generator_identity);
}

class ScriptedSamplePool final
    : public training::SamplePoolIngressService::Service {
public:
    enum class PushMode {
        kAccept,
        kUnavailableThenDuplicate,
        kAlwaysUnavailable,
    };

    ScriptedSamplePool(ContractConfig contract, PushMode mode)
        : contract_(std::move(contract)), mode_(mode) {}

    grpc::Status GetStatus(
        grpc::ServerContext*, const training::SamplePoolStatusReq*,
        training::SamplePoolStatusRsp* response) override {
        FillContract(contract_, response->mutable_contract());
        FillAuthority(response->mutable_sample_pool());
        response->set_ready(true);
        response->set_ingress_ready(true);
        response->set_pool_ready(true);
        response->set_backend_type(training::SAMPLE_BACKEND_TYPE_LOCAL_MEMORY);
        return grpc::Status::OK;
    }

    grpc::Status PushSamples(
        grpc::ServerContext*, const training::PushSamplesReq* request,
        training::PushSamplesRsp* response) override {
        const int call = ++push_calls_;
        std::string bytes;
        request->batch().SerializeToString(&bytes);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (first_batch_bytes_.empty()) {
                first_batch_bytes_ = bytes;
            } else if (first_batch_bytes_ != bytes) {
                exact_retry_ = false;
            }
        }

        if (mode_ == PushMode::kAlwaysUnavailable ||
            (mode_ == PushMode::kUnavailableThenDuplicate && call == 1)) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                "scripted SamplePool outage");
        }

        response->set_ret_code(0);
        response->set_batch_id(request->batch().batch_id());
        FillAuthority(response->mutable_sample_pool());
        response->set_pressure_state(training::PRESSURE_STATE_NORMAL);
        if (mode_ == PushMode::kUnavailableThenDuplicate) {
            response->set_result(training::PUSH_RESULT_DUPLICATE);
            response->set_accepted_samples(0);
            response->set_accepted_unique_samples(0);
        } else {
            response->set_result(training::PUSH_RESULT_ACCEPTED);
            response->set_accepted_samples(request->batch().samples_size());
            response->set_accepted_unique_samples(
                request->batch().samples_size());
        }
        return grpc::Status::OK;
    }

    int push_calls() const { return push_calls_.load(); }

    bool exact_retry() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return exact_retry_;
    }

private:
    static void FillAuthority(common::ServiceInstanceIdentity* identity) {
        identity->set_component("sample-pool");
        identity->set_instance_id("sample-pool-test");
        identity->set_lifecycle_epoch(1);
    }

    ContractConfig contract_;
    PushMode mode_;
    std::atomic<int> push_calls_{0};
    mutable std::mutex mutex_;
    std::string first_batch_bytes_;
    bool exact_retry_ = true;
};

struct RunningServer {
    RunningServer() = default;
    RunningServer(const RunningServer&) = delete;
    RunningServer& operator=(const RunningServer&) = delete;
    RunningServer(RunningServer&&) = default;
    RunningServer& operator=(RunningServer&&) = default;

    std::unique_ptr<grpc::Server> server;
    int port = 0;

    ~RunningServer() {
        if (server) {
            server->Shutdown();
            server->Wait();
        }
    }
};

RunningServer StartServer(ScriptedSamplePool& service) {
    grpc::ServerBuilder builder;
    RunningServer running;
    builder.AddListeningPort(
        "127.0.0.1:0", grpc::InsecureServerCredentials(), &running.port);
    builder.RegisterService(&service);
    running.server = builder.BuildAndStart();
    Require(running.server != nullptr && running.port > 0,
            "SamplePool ingress starts");
    return running;
}

AIServerConfig Config(int port) {
    AIServerConfig config;
    config.contract.source_digest.hex = std::string(64, 'a');
    config.contract.artifact_digest.hex = std::string(64, 'b');
    config.contract.generator_identity = std::string(64, 'c');
    config.task.agent_num = 1;
    config.sample_distributor.host = "127.0.0.1";
    config.sample_distributor.port = port;
    config.sample_distributor.health_timeout_ms = 500;
    config.sample_distributor.rpc_timeout_ms = 100;
    config.sample_distributor.max_attempts = 1;
    config.sample_distributor.enqueue_timeout_ms = 20;
    config.sample_distributor.drain_timeout_ms = 50;
    config.sample_distributor.status_poll_interval_ms = 20;
    config.sample_distributor.recovery_timeout_ms = 1000;
    config.sample_distributor.outbound_max_fragments = 8;
    config.sample_distributor.outbound_max_estimated_bytes = 1024 * 1024;
    return config;
}

training::SampleBatch Batch(const std::string& batch_id, char digest_byte) {
    training::SampleBatch batch;
    batch.set_batch_id(batch_id);
    batch.mutable_payload_digest()->set_algorithm(
        common::DIGEST_ALGORITHM_SHA256);
    batch.mutable_payload_digest()->set_hex(std::string(64, digest_byte));
    batch.set_created_at_unix_ms(UnixNowMs());
    batch.mutable_behavior_policy()->set_model_step(1);
    for (int index = 0; index < 2; ++index) {
        batch.add_samples()->set_reward(0.0f);
    }
    return batch;
}

bool WaitUntil(const std::function<bool()>& predicate, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return predicate();
}

void VerifyAcceptedPush() {
    AIServerConfig seed = Config(1);
    ScriptedSamplePool service(
        seed.contract, ScriptedSamplePool::PushMode::kAccept);
    auto running = StartServer(service);
    AIServerConfig config = Config(running.port);

    SampleDistributor distributor(config);
    Require(distributor.Start(), "SampleDistributor connects to SamplePool");
    Require(distributor.Enqueue(Batch("accepted-fragment", 'd')),
            "complete fragment enters the local outbound queue");
    Require(WaitUntil(
                [&]() {
                    const auto snapshot = distributor.GetSnapshot();
                    return snapshot.accepted_unique_batches == 1 &&
                           snapshot.queue_fragments == 0;
                },
                2000),
            "accepted fragment leaves the local outbound queue");
    Require(service.push_calls() == 1,
            "accepted fragment uses one ingress PushSamples call");
    Require(distributor.StopAndDrain(),
            "accepted distributor drains without an unresolved fragment");
}

void VerifyExactRetryAndRecovery() {
    AIServerConfig seed = Config(1);
    ScriptedSamplePool service(
        seed.contract,
        ScriptedSamplePool::PushMode::kUnavailableThenDuplicate);
    auto running = StartServer(service);
    AIServerConfig config = Config(running.port);

    SampleDistributor distributor(config);
    Require(distributor.Start(), "retry distributor connects to SamplePool");
    Require(distributor.Enqueue(Batch("retried-fragment", 'e')),
            "retry fragment enters the local outbound queue");
    Require(WaitUntil(
                [&]() {
                    const auto snapshot = distributor.GetSnapshot();
                    return snapshot.sample_delivery_paused &&
                           !snapshot.ready && snapshot.transient_retry;
                },
                1000),
            "UNAVAILABLE pauses delivery with readiness false");
    Require(WaitUntil(
                [&]() {
                    const auto snapshot = distributor.GetSnapshot();
                    return snapshot.duplicate_push_attempt_count == 1 &&
                           snapshot.queue_fragments == 0 && snapshot.ready;
                },
                2000),
            "same fragment recovers through DUPLICATE acceptance");
    Require(service.push_calls() == 2 && service.exact_retry(),
            "recovery retries the exact immutable fragment");
    Require(distributor.StopAndDrain(),
            "recovered distributor drains cleanly");
}

void VerifyRecoveryDeadline() {
    AIServerConfig seed = Config(1);
    ScriptedSamplePool service(
        seed.contract, ScriptedSamplePool::PushMode::kAlwaysUnavailable);
    auto running = StartServer(service);
    AIServerConfig config = Config(running.port);
    config.sample_distributor.recovery_timeout_ms = 40;

    SampleDistributor distributor(config);
    Require(distributor.Start(),
            "deadline distributor completes the initial health probe");
    Require(distributor.Enqueue(Batch("deadline-fragment", 'f')),
            "deadline fragment enters the local outbound queue");
    Require(WaitUntil(
                [&]() {
                    const auto snapshot = distributor.GetSnapshot();
                    return snapshot.sample_delivery_paused &&
                           !snapshot.ready && snapshot.transient_retry;
                },
                1000),
            "outage pauses only at the distributor boundary");
    Require(WaitUntil(
                [&]() {
                    return distributor.GetSnapshot().terminal_fault;
                },
                2000),
            "recovery deadline exhaustion becomes a terminal fault");
    Require(!distributor.StopAndDrain(),
            "unaccepted fragment cannot be reported as drained");
}

}  // namespace

int main() {
    VerifyAcceptedPush();
    VerifyExactRetryAndRecovery();
    VerifyRecoveryDeadline();
    return 0;
}
