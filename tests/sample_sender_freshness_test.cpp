#include "sample/sample_sender.h"

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

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
        response->mutable_distributor()->set_component("sample-distributor");
        response->mutable_distributor()->set_instance_id("test-distributor");
        response->mutable_distributor()->set_lifecycle_epoch(1);
        if (call == 0) {
            response->set_result(
                training::SAMPLE_CREDIT_RESULT_WAIT_INFLIGHT_LIMIT);
            response->set_retry_after_ms(5);
            response->set_message("wait for current demand window");
        } else {
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
        response->mutable_distributor()->set_component("sample-distributor");
        response->mutable_distributor()->set_instance_id("wait-distributor");
        response->mutable_distributor()->set_lifecycle_epoch(1);
        response->set_result(
            training::SAMPLE_CREDIT_RESULT_WAIT_INFLIGHT_LIMIT);
        response->set_retry_after_ms(7);
        response->set_message("wait for current demand window");
        return grpc::Status::OK;
    }

private:
    ContractConfig contract_;
};

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
    Require(sender.TrainingCapacityRetryAfterMs() == 7,
            "downstream retry interval was not preserved");

    sender.MarkDegraded("test shutdown");
    sender.StopAndDrain();
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
    batch.mutable_behavior_policy()->set_model_version(139);
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
    Require(snapshot.producer_stale_samples_by_model.at(139) == 8,
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
    return 0;
}
