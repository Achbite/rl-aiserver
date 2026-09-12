#pragma once
#include "proto/training/training.grpc.pb.h"
#include "task/protocol/training_namespaces.h"
#include <filesystem>
#include <chrono>
#include <condition_variable>
#include <future>
#include <thread>
#include <vector>
#include <grpcpp/grpcpp.h>
#include "task/metrics/metric_event_service.h"
#include "rl_sdk/task_client.h"
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace model_fixture {
inline rl_sdk::ClientOptions ClientOptions() {
    rl_sdk::ClientOptions options;
    options.identity.set_component("environment-client");
    options.identity.set_instance_id("test-client");
    options.identity.set_lifecycle_epoch(1);
    options.environment_instance_id = "test-environment";
    options.open_request_id = "test-open";
    return options;
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) throw std::runtime_error("open fixed ONNX fixture");
    std::ostringstream output;
    output << input.rdbuf();
    if (input.bad()) throw std::runtime_error("read fixed ONNX fixture");
    return output.str();
}

training::ModelArtifactManifest MakeManifest(
    const std::string& model_bytes) {
    training::ModelArtifactManifest manifest;
    auto* identity = manifest.mutable_identity();
    identity->set_model_lineage_id("lineage-fixed");
    identity->set_model_step(0);
    manifest.set_size_bytes(static_cast<int64_t>(model_bytes.size()));
    manifest.set_trained_samples(0);
    manifest.set_published_at_unix_ms(1700000000000);
    return manifest;
}

class FixedModelDistributor final
    : public training::ModelDistributorService::Service {
public:
    FixedModelDistributor(training::ModelArtifactManifest manifest,
                          std::string model_bytes)
        : manifest_(std::move(manifest)),
          model_bytes_(std::move(model_bytes)) {}

    grpc::Status GetModelDistributorStatus(
        grpc::ServerContext*,
        const training::ModelDistributorStatusReq*,
        training::ModelDistributorStatusRsp* response) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++io_calls_;
        response->set_ready(true);
        FillAuthority(response->mutable_distributor());
        *response->mutable_latest_model() = manifest_.identity();
        response->set_available_floor_model_step(0);
        response->set_latest_available_model_step(manifest_.identity().model_step());
        return grpc::Status::OK;
    }

    grpc::Status GetModelManifest(
        grpc::ServerContext*,
        const training::GetModelManifestReq* request,
        training::GetModelManifestRsp* response) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++io_calls_;
        if (request->requested_model().model_lineage_id() !=
                manifest_.identity().model_lineage_id() ||
            !request->requested_model().has_model_step() ||
            request->requested_model().model_step() != manifest_.identity().model_step()) {
            response->set_result(training::MODEL_LOOKUP_RESULT_NOT_FOUND);
            response->set_message("fixed model not found");
            return grpc::Status::OK;
        }
        response->set_result(training::MODEL_LOOKUP_RESULT_FOUND);
        *response->mutable_manifest() = manifest_;
        FillAuthority(response->mutable_distributor());
        response->set_available_floor_model_step(0);
        response->set_latest_available_model_step(manifest_.identity().model_step());
        return grpc::Status::OK;
    }

    grpc::Status DownloadModel(
        grpc::ServerContext*,
        const training::DownloadModelReq* request,
        grpc::ServerWriter<training::ModelChunk>* writer) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++io_calls_;
        if (request->requested_model().SerializeAsString() !=
            manifest_.identity().SerializeAsString()) {
            return grpc::Status(
                grpc::StatusCode::NOT_FOUND, "fixed model not found");
        }
        training::ModelChunk chunk;
        *chunk.mutable_model() = manifest_.identity();
        chunk.set_offset(0);
        chunk.set_data(model_bytes_);
        writer->Write(chunk);
        return grpc::Status::OK;
    }

    grpc::Status AckModel(grpc::ServerContext* context,
                          const training::AckModelReq* request,
                          training::AckModelRsp* response) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++io_calls_;
        ack_ = *request;
        acks_.push_back(*request);
        if (request->model().model_step() > 0)
            candidate_ack_times_.emplace_back(std::chrono::system_clock::now(), context->deadline());
        if (request->model().model_step() > 0 && uncertain_acks_ != 0) {
            if (uncertain_acks_ > 0) --uncertain_acks_;
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "ACK receipt unavailable");
        }
        response->set_result(training::MODEL_ACK_RESULT_APPLIED);
        FillAuthority(response->mutable_distributor());
        return grpc::Status::OK;
    }

    void SetCandidate(uint64_t step, std::string bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        model_bytes_ = std::move(bytes);
        manifest_.mutable_identity()->set_model_step(step);
        manifest_.set_size_bytes(static_cast<int64_t>(model_bytes_.size()));
    }

    void SetUncertainAcks(int count) { std::lock_guard<std::mutex> lock(mutex_); uncertain_acks_ = count; }
    int io_calls() const { std::lock_guard<std::mutex> lock(mutex_); return io_calls_; }
    std::vector<training::AckModelReq> acks() const { std::lock_guard<std::mutex> lock(mutex_); return acks_; }

    using AckTimes = std::vector<std::pair<std::chrono::system_clock::time_point, std::chrono::system_clock::time_point>>;
    AckTimes candidate_ack_times() const { std::lock_guard<std::mutex> lock(mutex_); return candidate_ack_times_; }
    training::AckModelReq ack() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ack_;
    }

private:
    static void FillAuthority(common::ServiceInstanceIdentity* identity) {
        identity->set_component("model-distributor");
        identity->set_instance_id("model-distributor-fixed");
        identity->set_lifecycle_epoch(1);
    }

    training::ModelArtifactManifest manifest_;
    std::string model_bytes_;
    mutable std::mutex mutex_;
    training::AckModelReq ack_;
    std::vector<training::AckModelReq> acks_;
    AckTimes candidate_ack_times_;
    int uncertain_acks_ = 0;
    int io_calls_ = 0;
};

class CapturingSamplePool final
    : public training::SamplePoolIngressService::Service {
public:
    grpc::Status GetStatus(
        grpc::ServerContext*,
        const training::SamplePoolStatusReq*,
        training::SamplePoolStatusRsp* response) override {
        FillAuthority(response->mutable_sample_pool());
        response->set_ready(true);
        response->set_ingress_ready(true);
        response->set_pool_ready(true);
        response->set_backend_type(
            training::SAMPLE_BACKEND_TYPE_LOCAL_MEMORY);
        return grpc::Status::OK;
    }

    grpc::Status PushSamples(
        grpc::ServerContext*,
        const training::PushSamplesReq* request,
        training::PushSamplesRsp* response) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            envelopes_.push_back(request->envelope());
        }
        response->set_result(training::PUSH_RESULT_ACCEPTED);
        response->set_envelope_id(request->envelope().envelope_id());
        response->set_pressure_state(training::PRESSURE_STATE_NORMAL);
        FillAuthority(response->mutable_sample_pool());
        condition_.notify_all();
        return grpc::Status::OK;
    }

    bool WaitForEnvelopeCount(std::size_t expected) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(
            lock, std::chrono::seconds(2),
            [&] { return envelopes_.size() >= expected; });
    }

    std::vector<training::ProcessedTransitionEnvelope> envelopes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return envelopes_;
    }

private:
    static void FillAuthority(common::ServiceInstanceIdentity* identity) {
        identity->set_component("sample-pool");
        identity->set_instance_id("sample-pool-fixed");
        identity->set_lifecycle_epoch(1);
    }

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::vector<training::ProcessedTransitionEnvelope> envelopes_;
};

class LocalServer {
public:
    explicit LocalServer(std::initializer_list<grpc::Service*> services) {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        for (auto* service : services) builder.RegisterService(service);
        server = builder.BuildAndStart();
        if (!server || port <= 0) throw std::runtime_error("start local test RPC service");
    }
    ~LocalServer() { server->Shutdown(); server->Wait(); }
    int port = 0;
    std::unique_ptr<grpc::Server> server;
};

// Acts only as the neighboring metric consumer while the producer shuts down.
// This settles the real source-final protocol rather than bypassing teardown.
template<class Stop>
bool FinishMetrics(Stop stop, MetricEventService& metrics, const common::ServiceInstanceIdentity& source) {
    auto shutdown = std::async(std::launch::async, stop);
    training::GetMetricBatchReq get;
    get.mutable_consumer()->set_component("learner");
    get.mutable_consumer()->set_instance_id("metric-test-consumer");
    get.mutable_consumer()->set_lifecycle_epoch(1);
    *get.mutable_cursor()->mutable_source() = source;
    get.set_max_events(100);
    get.set_max_bytes(1024 * 1024);
    get.set_wait_timeout_ms(50);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (shutdown.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("shutdown did not settle metric final ACK");
        training::GetMetricBatchRsp response;
        metrics.GetMetricBatch(nullptr, &get, &response);
        if (response.has_batch()) {
            training::AckMetricBatchReq ack;
            *ack.mutable_consumer() = get.consumer();
            *ack.mutable_cursor()->mutable_source() = response.producer();
            ack.mutable_cursor()->set_acknowledged_batch_sequence(response.batch().batch_sequence());
            ack.mutable_cursor()->set_acknowledged_event_sequence(response.batch().final_event_sequence());
            training::AckMetricBatchRsp result;
            metrics.AckMetricBatch(nullptr, &ack, &result);
            *get.mutable_cursor() = result.committed_cursor();
        }
    }
    return shutdown.get();
}

class TemporaryRoot {
public:
    TemporaryRoot()
        : path_(std::filesystem::temp_directory_path() /
                ("aiserver-model-update-" + std::to_string(::getpid()))) {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~TemporaryRoot() { std::filesystem::remove_all(path_); }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};


}  // namespace model_fixture
