#include "grpc/maze_service.h"

#include <grpcpp/grpcpp.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

struct MazeServiceUpdateTestAccess {
    static const std::string& ProducerInstanceId(
        const MazeServiceImpl& service) {
        return service.producer_instance_id_;
    }

    static uint64_t ProducerLifecycleEpoch(
        const MazeServiceImpl& service) {
        return service.producer_lifecycle_epoch_;
    }

    static SessionManager::Session* AddSession(MazeServiceImpl& service) {
        const std::string id = service.session_mgr_.CreateSession();
        return service.session_mgr_.GetSession(id);
    }

    static void MakeReady(MazeServiceImpl& service) {
        service.state_.store(training::AISERVER_STATE_READY);
        service.model_state_.store(training::MODEL_STATE_READY);
        service.last_error_.clear();
        std::lock_guard<std::mutex> lock(service.sample_sender_.mutex_);
        service.sample_sender_.accepting_ = true;
        service.sample_sender_.ready_ = true;
        service.sample_sender_.degraded_ = false;
        service.sample_sender_.SetDeliveryStateLocked(
            SampleSender::DeliveryState::kHealthy);
        service.sample_sender_.transient_retry_after_ms_ = 0;
        service.sample_sender_.training_capacity_wait_ = false;
        service.sample_sender_.training_capacity_retry_after_ms_ = 0;
        service.sample_sender_.last_error_.clear();
    }

    static void SetModel(MazeServiceImpl& service,
                         int version,
                         int64_t train_updates,
                         int64_t trained_samples) {
        service.model_manifest_.model_version = version;
        service.model_manifest_.sha256 = std::string(
            64, static_cast<char>('a' + version % 6));
        service.model_manifest_.model_lineage_id = "atomicity-fixture";
        service.model_manifest_.manifest_digest = std::string(64, 'b');
        service.model_manifest_.train_updates = train_updates;
        service.model_manifest_.trained_samples = trained_samples;
        auto* identity = service.model_manifest_.wire.mutable_identity();
        identity->set_model_lineage_id("atomicity-fixture");
        identity->set_model_version(static_cast<uint64_t>(version));
        identity->mutable_artifact_digest()->set_algorithm(
            common::DIGEST_ALGORITHM_SHA256);
        identity->mutable_artifact_digest()->set_hex(
            service.model_manifest_.sha256);
        identity->mutable_manifest_digest()->set_algorithm(
            common::DIGEST_ALGORITHM_SHA256);
        identity->mutable_manifest_digest()->set_hex(
            service.model_manifest_.manifest_digest);
        service.model_manifest_.wire.set_train_updates(train_updates);
        service.model_manifest_.wire.set_trained_samples(trained_samples);
        std::string error;
        if (!service.task_controller_.Initialize(
                2, service.ActiveModelIdentity(), 0, error)) {
            std::cerr << "TaskController init failed: " << error << std::endl;
            std::exit(1);
        }
    }

    static void SetModel(MazeServiceImpl& service) {
        SetModel(service, 0, 0, 0);
    }

    static void StageModel(MazeServiceImpl& service,
                           int version,
                           int64_t train_updates,
                           int64_t trained_samples,
                           const std::string& staged_path) {
        service.staged_model_manifest_.model_version = version;
        service.staged_model_manifest_.sha256 = std::string(64, 'c');
        service.staged_model_manifest_.model_lineage_id =
            "atomicity-fixture";
        service.staged_model_manifest_.manifest_digest =
            std::string(64, 'd');
        service.staged_model_manifest_.train_updates = train_updates;
        service.staged_model_manifest_.trained_samples = trained_samples;
        service.staged_model_manifest_.model_path = staged_path;
        auto* identity =
            service.staged_model_manifest_.wire.mutable_identity();
        identity->set_model_lineage_id("atomicity-fixture");
        identity->set_model_version(static_cast<uint64_t>(version));
        identity->mutable_artifact_digest()->set_algorithm(
            common::DIGEST_ALGORITHM_SHA256);
        identity->mutable_artifact_digest()->set_hex(std::string(64, 'c'));
        identity->mutable_manifest_digest()->set_algorithm(
            common::DIGEST_ALGORITHM_SHA256);
        identity->mutable_manifest_digest()->set_hex(std::string(64, 'd'));
        service.staged_model_manifest_.wire.set_train_updates(train_updates);
        service.staged_model_manifest_.wire.set_trained_samples(
            trained_samples);
        std::string error;
        if (!service.onnx_inferencer_.PrepareModel(
                staged_path, 17, 9, service.staged_prepared_model_, &error)) {
            std::cerr << "staged model prepare failed: " << error << std::endl;
            std::exit(1);
        }
    }

    static bool LoadActiveAndStage(
        MazeServiceImpl& service,
        const std::string& active_path,
        const std::string& staged_path,
        std::string& error) {
        service.model_manifest_.model_path = active_path;
        if (!service.onnx_inferencer_.LoadModel(
                active_path, 17, 9, &error)) {
            return false;
        }
        StageModel(service, 1, 0, 0, staged_path);
        return true;
    }

    static bool LoadActive(MazeServiceImpl& service,
                           const std::string& active_path,
                           std::string& error) {
        service.model_manifest_.model_path = active_path;
        return service.onnx_inferencer_.LoadModel(
            active_path, 17, 9, &error);
    }

    static bool AckPending(const MazeServiceImpl& service) {
        return service.model_ack_pending_;
    }
    static int64_t ActiveModelVersion(const MazeServiceImpl& service) {
        return service.model_manifest_.HasModelIdentity()
                   ? static_cast<int64_t>(service.model_manifest_.model_version)
                   : -1;
    }
    static int64_t ActiveTrainUpdates(const MazeServiceImpl& service) {
        return service.model_manifest_.train_updates;
    }
    static int64_t ActiveTrainedSamples(const MazeServiceImpl& service) {
        return service.model_manifest_.trained_samples;
    }
    static int64_t StagedModelVersion(const MazeServiceImpl& service) {
        return service.staged_model_manifest_.HasModelIdentity()
                   ? static_cast<int64_t>(
                         service.staged_model_manifest_.model_version)
                   : -1;
    }
    static int64_t ModelSwitchCount(const MazeServiceImpl& service) {
        return service.model_switch_count_;
    }
    static bool RetryAck(MazeServiceImpl& service) {
        return service.RetryPendingModelAck();
    }
    static bool LoadInitialModel(MazeServiceImpl& service) {
        return service.LoadInitialModel();
    }
    static training::ModelState ModelState(const MazeServiceImpl& service) {
        return service.model_state_.load();
    }
    static bool RunWatcherActivationAttempt(MazeServiceImpl& service) {
        return service.TryActivateStagedModelForWatcher();
    }
    static void StartWatcher(MazeServiceImpl& service) {
        service.StartModelWatcher();
    }
    static void StopWatcher(MazeServiceImpl& service) {
        service.StopModelWatcher();
    }
    static bool InferActive(MazeServiceImpl& service,
                            const std::vector<float>& observation,
                            std::vector<float>& logits,
                            float& value) {
        return service.onnx_inferencer_.Infer(
            observation, static_cast<int>(observation.size()),
            logits, value);
    }
    static uint64_t NextEpisodeId(const MazeServiceImpl& service) {
        return service.next_episode_id_.load();
    }
    static std::string ControllerJson(const MazeServiceImpl& service) {
        return service.task_controller_.ToJson();
    }
    static SingleMapTaskSnapshot ControllerSnapshot(
        const MazeServiceImpl& service) {
        return service.task_controller_.GetSnapshot();
    }
    static std::string ActionRng(const MazeServiceImpl& service) {
        std::ostringstream output;
        output << service.action_rng_;
        return output.str();
    }
    static void SetProducedAndTrained(MazeServiceImpl& service,
                                      int64_t samples) {
        SetRunProgress(service, 1, samples, samples);
    }
    static void SetRunProgress(MazeServiceImpl& service,
                               int model_version,
                               int64_t produced_samples,
                               int64_t trained_samples) {
        service.produced_unique_samples_ = produced_samples;
        service.model_manifest_.model_version = model_version;
        service.model_manifest_.train_updates = model_version;
        service.model_manifest_.trained_samples = trained_samples;
        service.model_manifest_.wire.mutable_identity()->set_model_version(
            static_cast<uint64_t>(model_version));
        service.model_manifest_.wire.set_train_updates(model_version);
        service.model_manifest_.wire.set_trained_samples(trained_samples);
    }
    static void BindPendingActionsToActiveModel(
        MazeServiceImpl& service,
        SessionManager::Session& session) {
        for (auto& item : session.agents) {
            auto& agent = item.second;
            agent.pending_model_version = service.model_manifest_.model_version;
            agent.pending_model_checksum = service.model_manifest_.sha256;
            agent.pending_model_lineage_id =
                service.model_manifest_.model_lineage_id;
            agent.pending_model_manifest_digest =
                service.model_manifest_.manifest_digest;
        }
    }
    static void SetManifestDigest(MazeServiceImpl& service,
                                  const std::string& digest) {
        service.model_manifest_.manifest_digest = digest;
    }
    static void InjectSenderFault(MazeServiceImpl& service,
                                  const std::string& error) {
        {
            std::lock_guard<std::mutex> lock(service.sample_sender_.mutex_);
            service.sample_sender_.degraded_ = true;
            service.sample_sender_.delivery_state_ =
                SampleSender::DeliveryState::kTerminalFault;
            service.sample_sender_.last_error_ = error;
        }
        service.state_.store(training::AISERVER_STATE_DEGRADED);
        service.last_error_ = error;
    }
    static void SetSenderTransient(MazeServiceImpl& service,
                                   const std::string& error,
                                   int retry_after_ms) {
        service.sample_sender_.MarkTransient(error, retry_after_ms);
    }
    static void SetSenderFlowWait(MazeServiceImpl& service,
                                  int retry_after_ms) {
        std::lock_guard<std::mutex> lock(service.sample_sender_.mutex_);
        service.sample_sender_.accepting_ = true;
        service.sample_sender_.ready_ = true;
        service.sample_sender_.degraded_ = false;
        service.sample_sender_.training_capacity_wait_ = true;
        service.sample_sender_.training_capacity_retry_after_ms_ =
            retry_after_ms;
        service.sample_sender_.capacity_wait_started_ =
            std::chrono::steady_clock::now();
        service.sample_sender_.SetDeliveryStateLocked(
            SampleSender::DeliveryState::kFlowWait);
        service.sample_sender_.last_error_.clear();
    }
    static void MarkSenderTransientAfterSeal(MazeServiceImpl& service) {
        service.sample_sender_.MarkTransient(
            "injected sender state race after frame transaction seal", 43);
    }
    static void SetSenderTerminal(MazeServiceImpl& service,
                                  const std::string& error) {
        std::lock_guard<std::mutex> lock(service.sample_sender_.mutex_);
        service.sample_sender_.accepting_ = true;
        service.sample_sender_.ready_ = true;
        service.sample_sender_.degraded_ = true;
        service.sample_sender_.delivery_state_ =
            SampleSender::DeliveryState::kTerminalFault;
        service.sample_sender_.transient_retry_after_ms_ = 0;
        service.sample_sender_.last_error_ = error;
    }
    static void SetSenderStopped(MazeServiceImpl& service) {
        std::lock_guard<std::mutex> lock(service.sample_sender_.mutex_);
        service.sample_sender_.accepting_ = false;
        service.sample_sender_.ready_ = false;
        service.sample_sender_.degraded_ = false;
        service.sample_sender_.SetDeliveryStateLocked(
            SampleSender::DeliveryState::kStopped);
        service.sample_sender_.transient_retry_after_ms_ = 0;
        service.sample_sender_.last_error_.clear();
    }
    static std::string LastError(const MazeServiceImpl& service) {
        return service.last_error_;
    }

    static int64_t ProducedSamples(const MazeServiceImpl& service) {
        return service.produced_unique_samples_;
    }
    static int64_t ProducedBatches(const MazeServiceImpl& service) {
        return service.produced_unique_batches_;
    }
    static uint64_t FragmentSequence(const MazeServiceImpl& service) {
        return service.next_fragment_seq_.load();
    }
    static SampleSender::Snapshot SenderSnapshot(
        const MazeServiceImpl& service) {
        return service.sample_sender_.GetSnapshot();
    }
    static bool StartSampleSender(MazeServiceImpl& service) {
        return service.sample_sender_.Start();
    }
    static bool EnqueueSample(MazeServiceImpl& service,
                              const training::SampleBatch& batch) {
        return service.sample_sender_.Enqueue(batch);
    }
    static training::AIServerState ServiceState(
        const MazeServiceImpl& service) {
        return service.state_.load();
    }
    static void SetPendingModelAck(MazeServiceImpl& service,
                                   const std::string& error) {
        common::ServiceInstanceIdentity authority;
        authority.set_component("model-distributor");
        authority.set_instance_id("shutdown-pending-ack-authority");
        authority.set_lifecycle_epoch(1);
        service.RecordPendingModelAck(
            service.model_manifest_, authority, error);
    }
    static void AddInconsistentShutdownCache(
        MazeServiceImpl& service,
        SessionManager::Session& session) {
        session.agent_sample_caches[0].emplace_back();
        service.produced_unique_samples_ = 1;
        service.produced_samples_by_model_.clear();
    }
};

namespace {

void FillAtomicityContract(const ContractConfig& config,
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

int64_t UnixNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

class ShutdownResponseLossDistributor final
    : public training::SampleDistributorService::Service {
public:
    explicit ShutdownResponseLossDistributor(ContractConfig contract)
        : contract_(std::move(contract)) {}

    grpc::Status GetStatus(
        grpc::ServerContext*, const training::DistributorStatusReq*,
        training::DistributorStatusRsp* response) override {
        FillAtomicityContract(contract_, response->mutable_contract());
        FillAuthority(response->mutable_distributor());
        response->set_ready(true);
        response->set_ingress_ready(true);
        return grpc::Status::OK;
    }

    grpc::Status AcquireSampleCredit(
        grpc::ServerContext*, const training::AcquireSampleCreditReq* request,
        training::SampleCreditGrant* response) override {
        response->set_ret_code(0);
        response->set_result(training::SAMPLE_CREDIT_RESULT_GRANTED);
        response->set_message("credit granted");
        response->set_request_id(request->request_id());
        response->set_credit_id(request->batch_id() + "-credit");
        response->set_demand_id("shutdown-response-loss-demand");
        response->set_demand_epoch(1);
        response->set_batch_id(request->batch_id());
        *response->mutable_payload_digest() = request->payload_digest();
        response->set_granted_samples(request->sample_count());
        response->set_granted_fragments(request->fragment_count());
        response->set_granted_estimated_bytes(request->estimated_bytes());
        response->set_expires_at_unix_ms(UnixNowMs() + 60000);
        response->set_pressure_state(training::PRESSURE_STATE_NORMAL);
        response->set_state(training::SAMPLE_CREDIT_STATE_RESERVED);
        FillAuthority(response->mutable_distributor());
        return grpc::Status::OK;
    }

    grpc::Status PushSamples(
        grpc::ServerContext*, const training::PushSamplesReq*,
        training::PushSamplesRsp*) override {
        ++push_calls_;
        return grpc::Status(
            grpc::StatusCode::UNAVAILABLE,
            "injected persistent PushSamples response loss");
    }

    int push_calls() const { return push_calls_.load(); }

private:
    static void FillAuthority(common::ServiceInstanceIdentity* authority) {
        authority->set_component("sample-distributor");
        authority->set_instance_id("shutdown-response-loss-distributor");
        authority->set_lifecycle_epoch(1);
    }

    ContractConfig contract_;
    std::atomic<int> push_calls_{0};
};

class AuthorityDistributor final
    : public training::ModelDistributorService::Service {
public:
    enum class Mode {
        LossAfterApply,
        SameAuthorityRejected,
        SameAuthorityApplied,
        ChangedAuthorityRejected,
        ChangedAuthorityApplied,
        InvalidAuthorityApplied,
        ContradictoryApplied,
        Unspecified,
    };

    explicit AuthorityDistributor(ContractConfig contract)
        : contract_(std::move(contract)) {}

    grpc::Status GetModelDistributorStatus(
        grpc::ServerContext*,
        const training::ModelDistributorStatusReq*,
        training::ModelDistributorStatusRsp* response) override {
        if (!status_available_.load()) {
            return grpc::Status(
                grpc::StatusCode::UNAVAILABLE,
                "injected authority probe failure");
        }
        response->mutable_distributor()->set_component("model-distributor");
        response->mutable_distributor()->set_instance_id("authority-a");
        response->mutable_distributor()->set_lifecycle_epoch(1);
        FillAtomicityContract(contract_, response->mutable_contract());
        if (wrong_status_contract_.load()) {
            response->mutable_contract()->set_package_version("wrong-version");
        }
        response->set_ready(true);
        return grpc::Status::OK;
    }

    grpc::Status AckModel(
        grpc::ServerContext*, const training::AckModelReq* request,
        training::AckModelRsp* response) override {
        ++ack_calls_;
        std::function<void()> hook;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_load_instance_id_ = request->load_instance_id();
            hook = std::move(ack_hook_);
        }
        if (hook) hook();
        const Mode mode = mode_.load();
        if (mode == Mode::LossAfterApply) {
            applied_.store(true);
            return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                "response lost after remote apply");
        }
        response->mutable_distributor()->set_component("model-distributor");
        if (mode != Mode::InvalidAuthorityApplied) {
            response->mutable_distributor()->set_instance_id(
                mode == Mode::ChangedAuthorityRejected ||
                        mode == Mode::ChangedAuthorityApplied
                    ? "authority-b"
                    : "authority-a");
            response->mutable_distributor()->set_lifecycle_epoch(1);
        }
        if (mode == Mode::SameAuthorityRejected ||
            mode == Mode::ChangedAuthorityRejected) {
            response->set_ret_code(-1);
            response->set_result(training::MODEL_ACK_RESULT_REJECTED);
            response->set_message("injected rejection");
        } else if (mode == Mode::ContradictoryApplied) {
            response->set_ret_code(-1);
            response->set_result(training::MODEL_ACK_RESULT_APPLIED);
            response->set_message("injected response contradiction");
        } else if (mode == Mode::Unspecified) {
            response->set_ret_code(-1);
            response->set_result(training::MODEL_ACK_RESULT_UNSPECIFIED);
            response->set_message("injected unspecified result");
        } else {
            applied_.store(true);
            response->set_ret_code(0);
            response->set_result(
                training::MODEL_ACK_RESULT_ALREADY_APPLIED);
            response->set_message("already applied");
        }
        return grpc::Status::OK;
    }

    void SetMode(Mode mode) { mode_.store(mode); }
    void SetStatusAvailable(bool available) {
        status_available_.store(available);
    }
    void SetWrongStatusContract(bool wrong) {
        wrong_status_contract_.store(wrong);
    }
    void SetAckHook(std::function<void()> hook) {
        std::lock_guard<std::mutex> lock(mutex_);
        ack_hook_ = std::move(hook);
    }
    bool applied() const { return applied_.load(); }
    int ack_calls() const { return ack_calls_.load(); }
    std::string last_load_instance_id() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_load_instance_id_;
    }

private:
    ContractConfig contract_;
    std::atomic<bool> applied_{false};
    std::atomic<bool> status_available_{true};
    std::atomic<bool> wrong_status_contract_{false};
    std::atomic<int> ack_calls_{0};
    std::atomic<Mode> mode_{Mode::LossAfterApply};
    mutable std::mutex mutex_;
    std::string last_load_instance_id_;
    std::function<void()> ack_hook_;
};

class InitialModelDistributor final
    : public training::ModelDistributorService::Service {
public:
    enum class AckMode {
        Applied,
        LoseFirstTransaction,
        LoseEveryTransaction,
    };

    InitialModelDistributor(training::ModelArtifactManifest manifest,
                            std::string model_bytes)
        : manifest_(std::move(manifest)),
          model_bytes_(std::move(model_bytes)) {}

    grpc::Status GetModelManifest(
        grpc::ServerContext*, const training::GetModelManifestReq*,
        training::GetModelManifestRsp* response) override {
        response->set_ret_code(0);
        response->mutable_manifest()->CopyFrom(manifest_);
        response->set_available_floor_model_version(
            manifest_.identity().model_version());
        response->set_latest_available_model_version(
            manifest_.identity().model_version());
        return grpc::Status::OK;
    }

    grpc::Status DownloadModel(
        grpc::ServerContext*, const training::DownloadModelReq* request,
        grpc::ServerWriter<training::ModelChunk>* writer) override {
        training::ModelChunk chunk;
        chunk.mutable_model()->CopyFrom(request->requested_model());
        chunk.set_offset(0);
        chunk.set_data(model_bytes_);
        writer->Write(chunk);
        return grpc::Status::OK;
    }

    grpc::Status GetModelDistributorStatus(
        grpc::ServerContext*,
        const training::ModelDistributorStatusReq*,
        training::ModelDistributorStatusRsp* response) override {
        const int call = ++status_calls_;
        if (!status_available_.load() &&
            (status_available_after_call_.load() <= 0 ||
             call < status_available_after_call_.load())) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                "injected initial authority outage");
        }
        response->mutable_distributor()->set_component("model-distributor");
        response->mutable_distributor()->set_instance_id("initial-authority");
        response->mutable_distributor()->set_lifecycle_epoch(1);
        response->mutable_contract()->CopyFrom(manifest_.contract());
        if (wrong_status_contract_.load()) {
            response->mutable_contract()->mutable_source_digest()->set_hex(
                std::string(64, 'f'));
        }
        response->set_ready(true);
        return grpc::Status::OK;
    }

    grpc::Status AckModel(
        grpc::ServerContext*, const training::AckModelReq* request,
        training::AckModelRsp* response) override {
        const int call = ++ack_calls_;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_load_instance_id_ = request->load_instance_id();
            last_aiserver_.CopyFrom(request->aiserver());
        }
        if ((ack_mode_.load() == AckMode::LoseFirstTransaction && call <= 3) ||
            ack_mode_.load() == AckMode::LoseEveryTransaction) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                "initial ACK response lost after apply");
        }
        response->mutable_distributor()->set_component("model-distributor");
        response->mutable_distributor()->set_instance_id("initial-authority");
        response->mutable_distributor()->set_lifecycle_epoch(1);
        response->set_ret_code(0);
        response->set_result(training::MODEL_ACK_RESULT_ALREADY_APPLIED);
        response->set_message("initial ACK confirmed");
        return grpc::Status::OK;
    }

    void SetStatusAvailable(bool available) {
        status_available_.store(available);
    }
    void SetStatusAvailableAfterCall(int call) {
        status_available_.store(false);
        status_available_after_call_.store(call);
    }
    void SetWrongStatusContract(bool wrong) {
        wrong_status_contract_.store(wrong);
    }
    void SetAckMode(AckMode mode) { ack_mode_.store(mode); }
    int status_calls() const { return status_calls_.load(); }
    int ack_calls() const { return ack_calls_.load(); }
    std::string last_load_instance_id() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_load_instance_id_;
    }
    common::ServiceInstanceIdentity last_aiserver() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_aiserver_;
    }

private:
    training::ModelArtifactManifest manifest_;
    std::string model_bytes_;
    std::atomic<bool> status_available_{true};
    std::atomic<bool> wrong_status_contract_{false};
    std::atomic<int> status_available_after_call_{0};
    std::atomic<AckMode> ack_mode_{AckMode::Applied};
    std::atomic<int> status_calls_{0};
    std::atomic<int> ack_calls_{0};
    mutable std::mutex mutex_;
    std::string last_load_instance_id_;
    common::ServiceInstanceIdentity last_aiserver_;
};

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void SetDigest(const std::string& hex, common::ContentDigest* digest) {
    digest->set_algorithm(common::DIGEST_ALGORITHM_SHA256);
    digest->set_hex(hex);
}

void SetSchema(const SchemaConfig& source,
               common::SchemaIdentity* schema) {
    schema->set_schema_id(source.schema_id);
    schema->set_schema_version(source.schema_version);
    SetDigest(source.canonical_digest.hex,
              schema->mutable_canonical_digest());
}

std::string DeterministicBytes(
    const google::protobuf::MessageLite& message) {
    std::string output(message.ByteSizeLong(), '\0');
    google::protobuf::io::ArrayOutputStream array(
        output.data(), static_cast<int>(output.size()));
    google::protobuf::io::CodedOutputStream coded(&array);
    coded.SetSerializationDeterministic(true);
    Require(message.SerializeToCodedStream(&coded),
            "fixture protobuf serializes deterministically");
    output.resize(static_cast<std::size_t>(coded.ByteCount()));
    return output;
}

std::string Sha256(const std::string& payload) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    Require(context != nullptr, "fixture SHA256 context allocated");
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    const bool ok =
        EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
        EVP_DigestUpdate(context, payload.data(), payload.size()) == 1 &&
        EVP_DigestFinal_ex(context, digest.data(), &digest_size) == 1;
    EVP_MD_CTX_free(context);
    Require(ok, "fixture SHA256 succeeds");
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    return output.str();
}

std::string ReadBytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    Require(input.good(), "initial model fixture opens");
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

AIServerConfig Config() {
    AIServerConfig config;
    config.server.run_mode = aiserver_mode::kTraining;
    config.task.agent_num = 2;
    config.model.expected_model_lineage_id = "atomicity-fixture";
    config.contract.source_digest.hex = std::string(64, '1');
    config.contract.artifact_digest.hex = std::string(64, '2');
    config.contract.generator_identity = "atomicity-fixture-generator";
    config.training_semantics.observation_schema = {
        "maze.observation.v3", 1, {"sha256", std::string(64, '3')}};
    config.training_semantics.action_schema = {
        "maze.action.v1", 1, {"sha256", std::string(64, '4')}};
    config.training_semantics.reward_schema = {
        "maze.reward.v4", 1, {"sha256", std::string(64, '5')}};
    config.training_semantics.semantics_digest.hex = std::string(64, '6');
    config.sample_output.fragment_samples = 128;
    config.sample_output.outbound_max_fragments = 8;
    config.sample_output.outbound_max_estimated_bytes = 1024 * 1024;
    config.sample_output.enqueue_timeout_ms = 10;
    config.metrics.event_schema = {
        "maze.metrics.v2", 2,
        {"sha256",
         "34622334da8d4aec593ad231eb0e7cf4465fdee0cbfa13a9ea0e6f864797df73"}};
    return config;
}

training::ModelArtifactManifest InitialManifest(
    const AIServerConfig& config,
    const std::string& model_bytes,
    uint64_t model_version = 1,
    int64_t train_updates = 0) {
    training::ModelArtifactManifest manifest;
    manifest.set_manifest_schema_version(1);
    auto* contract = manifest.mutable_contract();
    contract->set_package_name(config.contract.package_name);
    contract->set_package_version(config.contract.package_version);
    SetDigest(config.contract.source_digest.hex,
              contract->mutable_source_digest());
    SetDigest(config.contract.artifact_digest.hex,
              contract->mutable_artifact_digest());
    contract->set_platform(config.contract.platform);
    contract->set_generator_identity(config.contract.generator_identity);
    auto* identity = manifest.mutable_identity();
    identity->set_model_lineage_id(config.model.expected_model_lineage_id);
    identity->set_model_version(model_version);
    SetDigest(Sha256(model_bytes), identity->mutable_artifact_digest());
    SetSchema(config.training_semantics.observation_schema,
              manifest.mutable_observation_schema());
    SetSchema(config.training_semantics.action_schema,
              manifest.mutable_action_schema());
    manifest.set_model_architecture_id(config.model.model_architecture_id);
    manifest.set_tensor_dtype(config.model.tensor_dtype);
    manifest.add_input_shape(1);
    manifest.add_input_shape(config.model.expected_obs_dim);
    manifest.add_action_shape(1);
    manifest.add_action_shape(config.model.expected_action_dim);
    manifest.add_value_shape(1);
    manifest.add_value_shape(1);
    std::ostringstream version_name;
    version_name << std::setfill('0') << std::setw(6) << model_version;
    manifest.set_artifact_uri(
        "fixture://" + version_name.str() + "/SaveModel.onnx");
    manifest.set_model_file("SaveModel.onnx");
    manifest.set_size_bytes(static_cast<int64_t>(model_bytes.size()));
    manifest.set_seed(0);
    manifest.set_train_updates(train_updates);
    manifest.set_trained_samples(0);
    SetDigest(std::string(64, '7'),
              manifest.mutable_training_config_digest());
    auto* semantics = manifest.mutable_training_semantics();
    semantics->set_training_contract_id(
        config.training_semantics.training_contract_id);
    SetSchema(config.training_semantics.observation_schema,
              semantics->mutable_observation_schema());
    SetSchema(config.training_semantics.action_schema,
              semantics->mutable_action_schema());
    SetSchema(config.training_semantics.reward_schema,
              semantics->mutable_reward_schema());
    semantics->set_policy_distribution_schema_id(
        config.training_semantics.policy_distribution_schema_id);
    semantics->set_model_architecture_id(
        config.training_semantics.model_architecture_id);
    SetDigest(config.training_semantics.semantics_digest.hex,
              semantics->mutable_semantics_digest());
    manifest.set_published_at_unix_ms(1);
    manifest.set_ready(true);
    auto digest_source = manifest;
    digest_source.mutable_identity()->clear_manifest_digest();
    SetDigest(Sha256(DeterministicBytes(digest_source)),
              manifest.mutable_identity()->mutable_manifest_digest());
    return manifest;
}

class WatcherDistributor final
    : public training::ModelDistributorService::Service {
public:
    WatcherDistributor(std::vector<training::ModelArtifactManifest> manifests,
                       std::string model_bytes)
        : manifests_(std::move(manifests)),
          model_bytes_(std::move(model_bytes)) {}

    grpc::Status GetModelManifest(
        grpc::ServerContext*, const training::GetModelManifestReq* request,
        training::GetModelManifestRsp* response) override {
        uint64_t version = request->latest_in_lineage()
            ? manifests_.back().identity().model_version()
            : request->requested_model().model_version();
        const auto found = std::find_if(
            manifests_.begin(), manifests_.end(), [&](const auto& manifest) {
                return manifest.identity().model_version() == version;
            });
        if (found == manifests_.end()) {
            response->set_ret_code(-1);
            response->set_message("requested watcher model is unavailable");
            return grpc::Status::OK;
        }
        response->set_ret_code(0);
        *response->mutable_manifest() = *found;
        FillAuthority(response->mutable_distributor());
        response->set_available_floor_model_version(
            manifests_.front().identity().model_version());
        response->set_latest_available_model_version(
            manifests_.back().identity().model_version());
        return grpc::Status::OK;
    }

    grpc::Status DownloadModel(
        grpc::ServerContext*, const training::DownloadModelReq* request,
        grpc::ServerWriter<training::ModelChunk>* writer) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            download_versions_.push_back(
                request->requested_model().model_version());
        }
        training::ModelChunk chunk;
        *chunk.mutable_model() = request->requested_model();
        chunk.set_offset(0);
        chunk.set_data(model_bytes_);
        writer->Write(chunk);
        return grpc::Status::OK;
    }

    grpc::Status AckModel(
        grpc::ServerContext*, const training::AckModelReq*,
        training::AckModelRsp* response) override {
        ++ack_calls_;
        if (!ack_available_.load()) {
            return grpc::Status(
                grpc::StatusCode::UNAVAILABLE,
                "injected watcher ACK response loss");
        }
        response->set_ret_code(0);
        response->set_result(training::MODEL_ACK_RESULT_ALREADY_APPLIED);
        response->set_message("watcher ACK converged");
        FillAuthority(response->mutable_distributor());
        return grpc::Status::OK;
    }

    grpc::Status GetModelDistributorStatus(
        grpc::ServerContext*, const training::ModelDistributorStatusReq*,
        training::ModelDistributorStatusRsp* response) override {
        response->set_ready(true);
        FillAuthority(response->mutable_distributor());
        *response->mutable_contract() = manifests_.back().contract();
        *response->mutable_latest_model() = manifests_.back().identity();
        response->set_available_floor_model_version(
            manifests_.front().identity().model_version());
        response->set_latest_available_model_version(
            manifests_.back().identity().model_version());
        return grpc::Status::OK;
    }

    void SetAckAvailable(bool available) {
        ack_available_.store(available);
    }
    std::vector<uint64_t> DownloadVersions() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return download_versions_;
    }
    int AckCalls() const { return ack_calls_.load(); }

private:
    static void FillAuthority(common::ServiceInstanceIdentity* identity) {
        identity->set_component("model-distributor");
        identity->set_instance_id("watcher-distributor");
        identity->set_lifecycle_epoch(1);
    }

    std::vector<training::ModelArtifactManifest> manifests_;
    std::string model_bytes_;
    mutable std::mutex mutex_;
    std::vector<uint64_t> download_versions_;
    std::atomic<bool> ack_available_{false};
    std::atomic<int> ack_calls_{0};
};

SessionManager::AgentRuntime Agent() {
    SessionManager::AgentRuntime agent;
    agent.last_action = 0;
    agent.prev_grid_x = 2;
    agent.prev_grid_y = 0;
    agent.has_pending_action = true;
    agent.pending_action = 0;
    agent.pending_action_frame_id = 0;
    agent.pending_log_prob = -0.25f;
    agent.pending_value = 0.5f;
    agent.pending_model_version = 0;
    agent.pending_model_checksum = std::string(64, 'a');
    agent.pending_model_lineage_id = "atomicity-fixture";
    agent.pending_model_manifest_digest = std::string(64, 'b');
    agent.pending_obs.assign(17, 0.25f);
    agent.visited.insert(2);
    agent.episode_start_geodesic_distance = 2;
    agent.observation_grid_x = 2;
    agent.observation_grid_y = 0;
    agent.last_observation_frame_id = 0;
    return agent;
}

void ConfigureSession(SessionManager::Session& session) {
    session.lifecycle_epoch = 7;
    session.task.set_task_contract_id("atomic-task");
    session.task.set_task_id("atomic-task-id");
    session.task.set_task_revision(1);
    session.task_state = maze::TASK_STATE_TRAINING;
    session.session_state = maze::SESSION_STATE_EPISODE_ACTIVE;
    session.protocol_episode_state = maze::EPISODE_STATE_RUNNING;
    session.evaluation_state = maze::EVALUATION_STATE_INACTIVE;
    session.episode_state = SessionManager::EpisodeState::Active;
    session.current_episode_id = "episode-1";
    session.current_episode_mode = maze::EPISODE_MODE_TRAINING;
    session.behavior_policy_scope = BehaviorPolicyScope::TrainingFragment;
    session.last_frame_id = 0;
    session.current_max_steps = 1;
    session.grid_cols = 3;
    session.grid_rows = 1;
    session.start_gx = 2;
    session.start_gy = 0;
    session.end_gx = 0;
    session.end_gy = 0;
    session.shortest_action_steps = 2;
    session.curriculum_stage = maze::CURRICULUM_STAGE_8X;
    session.blocked.assign(3, false);
    session.geodesic_distance = {0, 1, 2};
    session.agents.emplace(0, Agent());
    session.agents.emplace(1, Agent());
}

void ConfigureIdleSession(SessionManager::Session& session) {
    session.lifecycle_epoch = 9;
    session.task.set_task_contract_id("atomic-task");
    session.task.set_task_id("atomic-task-id");
    session.task.set_task_revision(1);
    session.initialized = true;
    session.task_state = maze::TASK_STATE_TRAINING;
    session.session_state = maze::SESSION_STATE_IDLE;
    session.protocol_episode_state = maze::EPISODE_STATE_UNSPECIFIED;
    session.evaluation_state = maze::EVALUATION_STATE_INACTIVE;
    session.episode_state = SessionManager::EpisodeState::None;
    session.shortest_action_steps = 2;
    session.grid_cols = 3;
    session.grid_rows = 1;
    session.start_gx = 2;
    session.start_gy = 0;
    session.end_gx = 0;
    session.end_gy = 0;
    session.blocked.assign(3, false);
    session.geodesic_distance = {0, 1, 2};
    session.agents.emplace(0, SessionManager::AgentRuntime{});
    session.agents.emplace(1, SessionManager::AgentRuntime{});
}

maze::BeginEpisodeReq BeginRequest(const SessionManager::Session& session,
                                   const std::string& key) {
    maze::BeginEpisodeReq request;
    auto* command = request.mutable_command();
    command->mutable_task()->CopyFrom(session.task);
    command->set_session_id(session.session_id);
    command->set_lifecycle_epoch(session.lifecycle_epoch);
    command->set_command_sequence(session.last_command_sequence + 1);
    command->set_idempotency_key(key);
    command->set_expected_task_state(session.task_state);
    command->set_expected_session_state(session.session_state);
    command->set_expected_episode_state(session.protocol_episode_state);
    command->set_expected_evaluation_state(session.evaluation_state);
    return request;
}

const std::vector<float>& ModelProbeObservation() {
    static const std::vector<float> observation = {
        0.0f, 0.6931471824645996f, 1.3862943649291992f, 0.25f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };
    return observation;
}

bool Near(float lhs, float rhs, float tolerance = 1e-6f) {
    return std::abs(lhs - rhs) <= tolerance;
}

void RequireActiveV0(MazeServiceImpl& service,
                     const std::string& message) {
    std::vector<float> logits;
    float value = 0.0f;
    Require(MazeServiceUpdateTestAccess::InferActive(
                service, ModelProbeObservation(), logits, value) &&
                logits.size() == 9 && Near(logits[8], 4.0f) &&
                Near(logits[2], 0.0f) && Near(value, -0.75f),
            message);
}

void RequireActiveV1(MazeServiceImpl& service,
                     const std::string& message) {
    std::vector<float> logits;
    float value = 0.0f;
    Require(MazeServiceUpdateTestAccess::InferActive(
                service, ModelProbeObservation(), logits, value) &&
                logits.size() == 9 &&
                Near(logits[1], 0.6931471824645996f) &&
                Near(logits[2], 1.3862943649291992f) &&
                Near(logits[8], 0.0f) && Near(value, 0.25f),
            message);
}

struct FixturePaths {
    std::filesystem::path root;
    std::filesystem::path active;
    std::filesystem::path staged;
};

FixturePaths InstallDistinctModels(const std::string& tag,
                                   const std::string& active_fixture,
                                   const std::string& staged_fixture) {
    namespace fs = std::filesystem;
    FixturePaths paths;
    paths.root = fs::temp_directory_path() /
        ("a3-update-atomicity-" + tag + "-" + std::to_string(getpid()));
    fs::remove_all(paths.root);
    fs::create_directories(paths.root / "active");
    fs::create_directories(paths.root / "incoming");
    paths.active = paths.root / "active" / "model.onnx";
    paths.staged = paths.root / "incoming" / "model.onnx";
    fs::copy_file(active_fixture, paths.active);
    fs::copy_file(staged_fixture, paths.staged);
    return paths;
}

maze::UpdateReq Request(const SessionManager::Session& session,
                        bool second_agent_valid) {
    maze::UpdateReq request;
    auto* command = request.mutable_command();
    command->mutable_task()->CopyFrom(session.task);
    command->set_session_id(session.session_id);
    command->set_lifecycle_epoch(session.lifecycle_epoch);
    command->set_command_sequence(1);
    command->set_idempotency_key(
        second_agent_valid ? "atomic-valid" : "atomic-invalid");
    command->set_episode_id(session.current_episode_id);
    command->set_evaluation_id(session.current_evaluation_id);
    command->set_expected_task_state(session.task_state);
    command->set_expected_session_state(session.session_state);
    command->set_expected_episode_state(session.protocol_episode_state);
    command->set_expected_evaluation_state(session.evaluation_state);
    request.set_frame_id(1);
    for (int agent_id = 0; agent_id < 2; ++agent_id) {
        auto* state = request.add_agents();
        state->set_agent_id(static_cast<uint32_t>(agent_id));
        state->mutable_position()->set_x(
            agent_id == 1 && !second_agent_valid ? 3.0f : 2.0f);
        state->mutable_position()->set_y(0.0f);
        state->set_is_done(true);
        state->set_termination_reason(
            maze::MAZE_TERMINATION_REASON_TIME_LIMIT);
        state->set_last_move_blocked(false);
    }
    return request;
}

maze::UpdateReq ActiveRequest(
    const SessionManager::Session& session,
    uint64_t frame_id,
    const std::vector<int>& positions) {
    maze::UpdateReq request;
    auto* command = request.mutable_command();
    command->mutable_task()->CopyFrom(session.task);
    command->set_session_id(session.session_id);
    command->set_lifecycle_epoch(session.lifecycle_epoch);
    command->set_command_sequence(session.last_command_sequence + 1);
    command->set_idempotency_key(
        "active-frame-" + std::to_string(frame_id));
    command->set_episode_id(session.current_episode_id);
    command->set_evaluation_id(session.current_evaluation_id);
    command->set_expected_task_state(session.task_state);
    command->set_expected_session_state(session.session_state);
    command->set_expected_episode_state(session.protocol_episode_state);
    command->set_expected_evaluation_state(session.evaluation_state);
    request.set_frame_id(frame_id);
    for (int agent_id = 0; agent_id < 2; ++agent_id) {
        auto* state = request.add_agents();
        state->set_agent_id(static_cast<uint32_t>(agent_id));
        state->mutable_position()->set_x(
            static_cast<float>(positions[agent_id]));
        state->mutable_position()->set_y(0.0f);
        state->set_is_done(false);
        state->set_termination_reason(
            maze::MAZE_TERMINATION_REASON_ACTIVE);
        state->set_last_move_blocked(false);
    }
    return request;
}

void TestMultiAgentUpdateIsAtomic() {
    MazeServiceImpl service(Config());
    MazeServiceUpdateTestAccess::MakeReady(service);
    MazeServiceUpdateTestAccess::SetModel(service);
    auto* session = MazeServiceUpdateTestAccess::AddSession(service);
    Require(session != nullptr, "test session allocated");
    ConfigureSession(*session);

    const auto agent_zero_before = session->agents.at(0);
    const auto agent_one_before = session->agents.at(1);
    const auto invalid = Request(*session, false);
    maze::UpdateRsp rejected;
    service.Update(nullptr, &invalid, &rejected);
    Require(rejected.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_REJECTED,
            "invalid second Agent rejects the whole Update");
    const auto& agent_zero = session->agents.at(0);
    const auto& agent_one = session->agents.at(1);
    Require(session->last_frame_id == 0,
            "rejected Update does not advance last_frame");
    Require(session->last_command_sequence == 0,
            "rejected Update does not commit lifecycle sequence");
    Require(agent_zero.has_pending_action &&
                agent_zero.pending_action_frame_id ==
                    agent_zero_before.pending_action_frame_id &&
                agent_zero.pending_obs == agent_zero_before.pending_obs,
            "earlier Agent pending action is unchanged");
    Require(agent_zero.last_observation_frame_id ==
                agent_zero_before.last_observation_frame_id &&
                agent_zero.visited == agent_zero_before.visited &&
                agent_zero.observation_done ==
                    agent_zero_before.observation_done,
            "earlier Agent ApplyState effects are discarded");
    Require(agent_zero.episode_return == 0.0 &&
                agent_zero.episode_transition_count == 0 &&
                agent_zero.reward_component_sums.empty(),
            "earlier Agent Reward accounting is discarded");
    Require(agent_one.has_pending_action ==
                agent_one_before.has_pending_action,
            "later Agent state is unchanged");
    Require(session->agent_sample_caches.empty(),
            "rejected Update exposes no sample cache prefix");
    Require(MazeServiceUpdateTestAccess::ProducedSamples(service) == 0 &&
                MazeServiceUpdateTestAccess::ProducedBatches(service) == 0 &&
                MazeServiceUpdateTestAccess::FragmentSequence(service) == 0,
            "rejected Update exposes no sample or fragment counters");
    Require(MazeServiceUpdateTestAccess::SenderSnapshot(service)
                    .queue_fragments == 0,
            "rejected Update enqueues no fragment");

    const auto valid = Request(*session, true);
    maze::UpdateRsp applied;
    service.Update(nullptr, &valid, &applied);
    Require(applied.lifecycle().result() == maze::LIFECYCLE_RESULT_APPLIED,
            "corrected frame applies once");
    Require(session->last_frame_id == 1 &&
                session->last_command_sequence == 1,
            "corrected frame and lifecycle sequence commit together");
    Require(session->protocol_episode_state ==
                maze::EPISODE_STATE_TERMINAL_REPORTED,
            "corrected terminal frame commits terminal state");
    for (const auto& item : session->agents) {
        const auto& agent = item.second;
        Require(!agent.has_pending_action && agent.done_collected &&
                    agent.episode_transition_count == 1 &&
                    agent.episode_behavior_model_seen &&
                    agent.episode_behavior_model_version_min == 0 &&
                    agent.episode_behavior_model_version_max == 0 &&
                    agent.episode_behavior_model_lineage_id ==
                        "atomicity-fixture",
                "each Agent transition commits exactly once");
    }
    Require(MazeServiceUpdateTestAccess::ProducedSamples(service) == 2 &&
                MazeServiceUpdateTestAccess::ProducedBatches(service) == 2 &&
                MazeServiceUpdateTestAccess::FragmentSequence(service) == 2,
            "corrected Update commits balanced sample counters");
    const auto sender =
        MazeServiceUpdateTestAccess::SenderSnapshot(service);
    Require(sender.queue_fragments == 2 && sender.queue_samples == 2,
            "complete two-Agent batch set becomes visible atomically");
}

void TestStandaloneEvaluationDoesNotRunTrainingReward() {
    auto config = Config();
    config.server.run_mode = aiserver_mode::kModelEvaluation;
    MazeServiceImpl service(config);
    MazeServiceUpdateTestAccess::MakeReady(service);
    MazeServiceUpdateTestAccess::SetModel(service);
    auto* session = MazeServiceUpdateTestAccess::AddSession(service);
    Require(session != nullptr, "standalone evaluation session allocated");
    ConfigureSession(*session);
    session->task_state = maze::TASK_STATE_EVALUATING;
    session->current_episode_mode = maze::EPISODE_MODE_EVALUATION_ARGMAX;
    session->behavior_policy_scope = BehaviorPolicyScope::EvaluationEpisode;
    session->current_evaluation_id = "standalone-evaluation-1";
    session->evaluation_state = maze::EVALUATION_STATE_ARGMAX_ROUND_1;
    session->evaluation_pinned_model_version = 0;
    session->evaluation_pinned_model_checksum = std::string(64, 'a');
    session->evaluation_pinned_model_lineage_id = "atomicity-fixture";
    session->evaluation_pinned_model_manifest_digest = std::string(64, 'b');

    auto active_at_horizon = Request(*session, true);
    active_at_horizon.mutable_command()->set_idempotency_key(
        "evaluation-active-at-horizon");
    for (auto& state : *active_at_horizon.mutable_agents()) {
        state.set_is_done(false);
        state.set_termination_reason(
            maze::MAZE_TERMINATION_REASON_ACTIVE);
    }
    maze::UpdateRsp horizon_rejected;
    service.Update(nullptr, &active_at_horizon, &horizon_rejected);
    Require(horizon_rejected.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_REJECTED &&
                session->last_command_sequence == 0 &&
                session->last_frame_id == 0,
            "standalone evaluation cannot advance beyond its horizon");

    auto terminal = Request(*session, true);
    terminal.mutable_command()->set_idempotency_key(
        "evaluation-terminal-at-horizon");
    maze::UpdateRsp applied;
    service.Update(nullptr, &terminal, &applied);
    Require(applied.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED,
            "standalone evaluation terminal transition applies");
    for (const auto& item : session->agents) {
        const auto& agent = item.second;
        Require(agent.episode_transition_count == 1 &&
                    agent.episode_return == 0.0 &&
                    agent.reward_component_sums.empty(),
                "standalone evaluation executed training Reward V4");
    }
    Require(MazeServiceUpdateTestAccess::ProducedSamples(service) == 0 &&
                MazeServiceUpdateTestAccess::ProducedBatches(service) == 0 &&
                MazeServiceUpdateTestAccess::SenderSnapshot(service)
                        .queue_samples == 0,
            "standalone evaluation entered the training sample pipeline");
}

void TestUpdateCapabilityReadinessAndErrors(
    const std::string& active_fixture) {
    {
        MazeServiceImpl service(Config());
        MazeServiceUpdateTestAccess::MakeReady(service);
        MazeServiceUpdateTestAccess::SetModel(service);
        auto* session = MazeServiceUpdateTestAccess::AddSession(service);
        Require(session != nullptr,
                "training readiness session allocated");
        ConfigureSession(*session);
        const auto agent_before = session->agents.at(0);
        MazeServiceUpdateTestAccess::SetSenderTransient(
            service, "injected Acquire deadline", 41);

        const auto request = Request(*session, true);
        maze::UpdateRsp response;
        service.Update(nullptr, &request, &response);
        Require(response.lifecycle().result() ==
                    maze::LIFECYCLE_RESULT_WAIT &&
                    response.lifecycle().message().find(
                        "sample delivery is recovering") !=
                        std::string::npos &&
                    response.environment_control() ==
                        maze::ENVIRONMENT_CONTROL_WAIT_FOR_TRAINING_CAPACITY &&
                    response.retry_after_ms() == 41 &&
                    response.actions_size() == 0,
                "training Update exposes an explicit retryable sample "
                "delivery wait");
        Require(session->last_frame_id == 0 &&
                    session->last_command_sequence == 0 &&
                    session->agents.at(0).pending_obs ==
                        agent_before.pending_obs &&
                    session->agents.at(0).episode_transition_count == 0 &&
                    MazeServiceUpdateTestAccess::ProducedSamples(service) ==
                        0 &&
                    MazeServiceUpdateTestAccess::ProducedBatches(service) ==
                        0,
                "training delivery WAIT advanced frame, lifecycle, Reward, "
                "or sample state");

        MazeServiceUpdateTestAccess::SetSenderTerminal(
            service, "injected contract mismatch");
        maze::UpdateRsp terminal;
        service.Update(nullptr, &request, &terminal);
        Require(terminal.lifecycle().result() ==
                    maze::LIFECYCLE_RESULT_REJECTED &&
                    terminal.lifecycle().message().find(
                        "sample delivery has a terminal fault") !=
                        std::string::npos &&
                    terminal.lifecycle().message().find(
                        "active episode") == std::string::npos &&
                    session->last_frame_id == 0 &&
                    session->last_command_sequence == 0,
                "terminal delivery error was collapsed into episode state");

        MazeServiceUpdateTestAccess::MakeReady(service);
        maze::UpdateRsp recovered;
        service.Update(nullptr, &request, &recovered);
        Require(recovered.lifecycle().result() ==
                    maze::LIFECYCLE_RESULT_APPLIED &&
                    session->last_frame_id == 1 &&
                    session->last_command_sequence == 1 &&
                    MazeServiceUpdateTestAccess::ProducedSamples(service) ==
                        2 &&
                    MazeServiceUpdateTestAccess::ProducedBatches(service) ==
                        2,
                "the identical WAITed command applies once after delivery "
                "recovery");
    }

    {
        MazeServiceImpl service(Config());
        MazeServiceUpdateTestAccess::MakeReady(service);
        MazeServiceUpdateTestAccess::SetModel(service);
        auto* session = MazeServiceUpdateTestAccess::AddSession(service);
        Require(session != nullptr, "inactive error session allocated");
        ConfigureSession(*session);
        session->episode_state = SessionManager::EpisodeState::Ended;
        const auto request = ActiveRequest(*session, 1, {2, 2});
        maze::UpdateRsp response;
        service.Update(nullptr, &request, &response);
        Require(response.lifecycle().result() ==
                    maze::LIFECYCLE_RESULT_REJECTED &&
                    response.lifecycle().message() ==
                        "Update requires an active episode",
                "inactive episode has a distinct Update rejection reason");
    }
}

void TestBeginAllowsRecoverableSenderStates() {
    {
        MazeServiceImpl service(Config());
        MazeServiceUpdateTestAccess::MakeReady(service);
        MazeServiceUpdateTestAccess::SetModel(service);
        auto* session = MazeServiceUpdateTestAccess::AddSession(service);
        Require(session != nullptr, "flow-wait training Begin session allocated");
        ConfigureIdleSession(*session);
        MazeServiceUpdateTestAccess::SetSenderFlowWait(service, 47);

        const auto begin_request =
            BeginRequest(*session, "begin-training-during-flow-wait");
        maze::BeginEpisodeRsp begin_response;
        service.BeginEpisode(nullptr, &begin_request, &begin_response);
        Require(begin_response.lifecycle().result() ==
                    maze::LIFECYCLE_RESULT_APPLIED &&
                    begin_response.assignment().mode() ==
                        maze::EPISODE_MODE_TRAINING &&
                    begin_response.assignment().collect_training_samples() &&
                    session->last_command_sequence == 1,
                "training episode assignment survives recoverable flow wait");

        const auto update_request = ActiveRequest(*session, 0, {2, 2});
        maze::UpdateRsp update_response;
        service.Update(nullptr, &update_request, &update_response);
        Require(update_response.lifecycle().result() ==
                    maze::LIFECYCLE_RESULT_WAIT &&
                    update_response.environment_control() ==
                        maze::ENVIRONMENT_CONTROL_WAIT_FOR_TRAINING_CAPACITY &&
                    update_response.retry_after_ms() == 47 &&
                    session->last_frame_id == -1 &&
                    session->last_command_sequence == 1,
                "first training Update freezes on flow wait without consuming "
                "the frame");
    }

    {
        MazeServiceImpl service(Config());
        MazeServiceUpdateTestAccess::MakeReady(service);
        MazeServiceUpdateTestAccess::SetModel(service);
        auto* session = MazeServiceUpdateTestAccess::AddSession(service);
        Require(session != nullptr, "terminal Begin session allocated");
        ConfigureIdleSession(*session);
        MazeServiceUpdateTestAccess::SetSenderTerminal(
            service, "injected terminal delivery fault");

        const auto request =
            BeginRequest(*session, "begin-during-terminal-delivery");
        maze::BeginEpisodeRsp response;
        service.BeginEpisode(nullptr, &request, &response);
        Require(response.lifecycle().result() ==
                    maze::LIFECYCLE_RESULT_REJECTED &&
                    response.assignment().ByteSizeLong() == 0 &&
                    session->episode_state ==
                        SessionManager::EpisodeState::None &&
                    session->last_command_sequence == 0,
                "terminal sample delivery remains fail-closed at Begin");
    }
}

void TestResumeUpdateRejectsCounterRollbackAndAcceptsRepublish(
    const std::string& active_fixture,
    const std::string& staged_fixture) {
    AuthorityDistributor distributor(Config().contract);
    distributor.SetMode(AuthorityDistributor::Mode::SameAuthorityApplied);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&distributor);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "resume activation distributor starts");

    const auto paths = InstallDistinctModels(
        "resume-progress", active_fixture, staged_fixture);
    auto config = Config();
    config.model.local_train_dir = paths.root.string();
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    config.model_distribution.rpc_timeout_ms = 50;
    MazeServiceImpl service(config);
    MazeServiceUpdateTestAccess::MakeReady(service);
    MazeServiceUpdateTestAccess::SetModel(service, 201, 200, 10000);
    std::string error;
    Require(MazeServiceUpdateTestAccess::LoadActive(
                service, paths.active.string(), error),
            "resume active model loads: " + error);
    MazeServiceUpdateTestAccess::StageModel(
        service, 202, 199, 10000, paths.staged.string());
    auto* session = MazeServiceUpdateTestAccess::AddSession(service);
    Require(session != nullptr, "resume Update session allocated");
    ConfigureSession(*session);
    MazeServiceUpdateTestAccess::BindPendingActionsToActiveModel(
        service, *session);
    const auto request = Request(*session, true);
    const auto agent_zero_before = session->agents.at(0);
    const std::string controller_before =
        MazeServiceUpdateTestAccess::ControllerJson(service);
    const std::string rng_before =
        MazeServiceUpdateTestAccess::ActionRng(service);

    maze::UpdateRsp train_rollback;
    service.Update(nullptr, &request, &train_rollback);
    Require(train_rollback.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_REJECTED &&
                distributor.ack_calls() == 0 &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) ==
                    201 &&
                MazeServiceUpdateTestAccess::StagedModelVersion(service) ==
                    202 &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(service) == 0 &&
                std::filesystem::is_regular_file(paths.active) &&
                std::filesystem::is_regular_file(paths.staged),
            "train-update rollback is caught before Promote or ACK");
    Require(session->last_frame_id == 0 &&
                session->last_command_sequence == 0 &&
                session->agents.at(0).pending_obs ==
                    agent_zero_before.pending_obs &&
                MazeServiceUpdateTestAccess::ProducedSamples(service) == 0 &&
                MazeServiceUpdateTestAccess::ProducedBatches(service) == 0 &&
                MazeServiceUpdateTestAccess::FragmentSequence(service) == 0 &&
                MazeServiceUpdateTestAccess::ControllerJson(service) ==
                    controller_before &&
                MazeServiceUpdateTestAccess::ActionRng(service) == rng_before &&
                MazeServiceUpdateTestAccess::SenderSnapshot(service)
                        .queue_fragments == 0 &&
                !service.IsReady() &&
                MazeServiceUpdateTestAccess::LastError(service).find(
                    "staged model progress is invalid") != std::string::npos,
            "train-update rollback preserves all candidate business state");
    RequireActiveV0(service,
                    "train-update rollback keeps serving the active artifact");

    MazeServiceUpdateTestAccess::MakeReady(service);
    MazeServiceUpdateTestAccess::StageModel(
        service, 202, 200, 9999, paths.staged.string());
    maze::UpdateRsp sample_rollback;
    service.Update(nullptr, &request, &sample_rollback);
    Require(sample_rollback.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_REJECTED &&
                distributor.ack_calls() == 0 &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) ==
                    201 &&
                MazeServiceUpdateTestAccess::StagedModelVersion(service) ==
                    202 &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(service) == 0 &&
                MazeServiceUpdateTestAccess::ProducedSamples(service) == 0 &&
                MazeServiceUpdateTestAccess::ControllerJson(service) ==
                    controller_before &&
                session->last_command_sequence == 0 &&
                std::filesystem::is_regular_file(paths.active) &&
                std::filesystem::is_regular_file(paths.staged),
            "trained-sample rollback is caught atomically before ACK");

    MazeServiceUpdateTestAccess::MakeReady(service);
    MazeServiceUpdateTestAccess::StageModel(
        service, 202, 200, 10000, paths.staged.string());
    distributor.SetAckHook([&service]() {
        MazeServiceUpdateTestAccess::MarkSenderTransientAfterSeal(service);
    });
    maze::UpdateRsp republished;
    service.Update(nullptr, &request, &republished);
    const auto controller =
        MazeServiceUpdateTestAccess::ControllerSnapshot(service);
    const auto sender_after_switch =
        MazeServiceUpdateTestAccess::SenderSnapshot(service);
    Require(republished.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED &&
                distributor.ack_calls() == 1 && !service.IsReady() &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) ==
                    202 &&
                MazeServiceUpdateTestAccess::ActiveTrainUpdates(service) ==
                    200 &&
                MazeServiceUpdateTestAccess::ActiveTrainedSamples(service) ==
                    10000 &&
                MazeServiceUpdateTestAccess::StagedModelVersion(service) < 0 &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(service) == 1,
            "higher publication with unchanged training counters activates");
    Require(session->last_frame_id == 1 &&
                session->last_command_sequence == 1 &&
                MazeServiceUpdateTestAccess::ProducedSamples(service) == 2 &&
                controller.startup_mode ==
                    SingleMapTaskStartupMode::Resume &&
                controller.baseline_model_version == 201 &&
                controller.baseline_train_updates == 200 &&
                controller.baseline_trained_samples == 10000 &&
                controller.run_produced_samples == 2 &&
                sender_after_switch.queue_fragments == 2 &&
                sender_after_switch.queue_samples == 2 &&
                sender_after_switch.transient_retry &&
                !sender_after_switch.degraded,
            "republish rolls model, frame, and the sealed sample set forward "
            "across a sender state race");
    RequireActiveV1(service,
                    "valid republish switches to the staged artifact");

    server->Shutdown();
    server->Wait();
    std::filesystem::remove_all(paths.root);
}

void TestLoadInitialAuthorityPreflightIsRetryable(
    const std::string& staged_fixture) {
    namespace fs = std::filesystem;
    const std::string model_bytes = ReadBytes(staged_fixture);
    auto config = Config();
    const fs::path root = fs::temp_directory_path() /
        ("a3-initial-preflight-" + std::to_string(getpid()));
    fs::remove_all(root);
    config.model.local_train_dir = root.string();
    config.model.startup_timeout_ms = 250;
    config.model_distribution.rpc_timeout_ms = 25;
    InitialModelDistributor distributor(
        InitialManifest(config, model_bytes), model_bytes);
    distributor.SetStatusAvailable(false);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&distributor);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "initial preflight distributor starts");
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    MazeServiceImpl service(config);

    Require(!MazeServiceUpdateTestAccess::LoadInitialModel(service),
            "persistent initial authority outage defers startup");
    const fs::path cached = root / "cache" / "000001" /
                            "SaveModel.onnx";
    Require(distributor.status_calls() > 0 && distributor.ack_calls() == 0 &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) < 0 &&
                MazeServiceUpdateTestAccess::ModelState(service) ==
                    training::MODEL_STATE_FAILED &&
                fs::is_regular_file(cached),
            "initial preflight failure sends no ACK but retains the complete "
            "non-active cache entry");

    server->Shutdown();
    server->Wait();
    fs::remove_all(root);

    auto retry_config = Config();
    const fs::path retry_root = fs::temp_directory_path() /
        ("a3-initial-preflight-retry-" + std::to_string(getpid()));
    fs::remove_all(retry_root);
    retry_config.model.local_train_dir = retry_root.string();
    retry_config.model.startup_timeout_ms = 1000;
    retry_config.model_distribution.rpc_timeout_ms = 25;
    InitialModelDistributor retry_distributor(
        InitialManifest(retry_config, model_bytes), model_bytes);
    retry_distributor.SetStatusAvailableAfterCall(2);
    grpc::ServerBuilder retry_builder;
    int retry_port = 0;
    retry_builder.AddListeningPort("127.0.0.1:0",
                                   grpc::InsecureServerCredentials(),
                                   &retry_port);
    retry_builder.RegisterService(&retry_distributor);
    std::unique_ptr<grpc::Server> retry_server =
        retry_builder.BuildAndStart();
    Require(retry_server != nullptr && retry_port > 0,
            "initial preflight recovery distributor starts");
    retry_config.model_distribution.host = "127.0.0.1";
    retry_config.model_distribution.port = retry_port;
    MazeServiceImpl retry_service(retry_config);

    Require(MazeServiceUpdateTestAccess::LoadInitialModel(retry_service),
            "one production initial load retries a transient authority outage");
    Require(retry_distributor.status_calls() >= 2 &&
                retry_distributor.ack_calls() == 1 &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(
                    retry_service) == 1 &&
                MazeServiceUpdateTestAccess::ModelState(retry_service) ==
                    training::MODEL_STATE_READY &&
                MazeServiceUpdateTestAccess::LastError(
                    retry_service).empty() &&
                fs::is_regular_file(
                    retry_root / "cache" / "000001" /
                    "SaveModel.onnx"),
            "recovered initial preflight activates and ACKs exactly once");
    RequireActiveV1(
        retry_service,
        "recovered initial preflight serves the tracked v1 model");

    retry_server->Shutdown();
    retry_server->Wait();
    fs::remove_all(retry_root);
}

void TestLoadInitialWrongContractIsRetryable(
    const std::string& staged_fixture) {
    namespace fs = std::filesystem;
    const std::string model_bytes = ReadBytes(staged_fixture);
    auto config = Config();
    const fs::path root = fs::temp_directory_path() /
        ("a3-initial-wrong-contract-" + std::to_string(getpid()));
    fs::remove_all(root);
    config.model.local_train_dir = root.string();
    config.model.startup_timeout_ms = 250;
    config.model_distribution.rpc_timeout_ms = 25;
    InitialModelDistributor distributor(
        InitialManifest(config, model_bytes), model_bytes);
    distributor.SetWrongStatusContract(true);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&distributor);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "initial wrong-contract distributor starts");
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    MazeServiceImpl service(config);

    const fs::path cached = root / "cache" / "000001" /
                            "SaveModel.onnx";
    Require(!MazeServiceUpdateTestAccess::LoadInitialModel(service),
            "wrong distributor contract blocks initial activation");
    Require(distributor.status_calls() > 0 && distributor.ack_calls() == 0 &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) < 0 &&
                MazeServiceUpdateTestAccess::ModelState(service) ==
                    training::MODEL_STATE_FAILED &&
                MazeServiceUpdateTestAccess::LastError(service).find(
                    "contract") != std::string::npos &&
                fs::is_regular_file(cached),
            "wrong initial contract performs no ACK, activation, or READY");

    distributor.SetWrongStatusContract(false);
    Require(MazeServiceUpdateTestAccess::LoadInitialModel(service),
            "corrected distributor contract retries the same initial model");
    Require(distributor.ack_calls() == 1 &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) == 1 &&
                MazeServiceUpdateTestAccess::ModelState(service) ==
                    training::MODEL_STATE_READY &&
                MazeServiceUpdateTestAccess::LastError(service).empty() &&
                fs::is_regular_file(cached),
            "corrected initial contract activates and ACKs exactly once");
    RequireActiveV1(service,
                    "corrected initial contract serves tracked v1");

    server->Shutdown();
    server->Wait();
    fs::remove_all(root);
}

void TestLoadInitialResponseLossConverges(
    const std::string& staged_fixture) {
    namespace fs = std::filesystem;
    const std::string model_bytes = ReadBytes(staged_fixture);
    auto config = Config();
    const fs::path root = fs::temp_directory_path() /
        ("a3-initial-response-loss-" + std::to_string(getpid()));
    fs::remove_all(root);
    config.model.local_train_dir = root.string();
    config.model.startup_timeout_ms = 1000;
    config.model_distribution.rpc_timeout_ms = 25;
    InitialModelDistributor distributor(
        InitialManifest(config, model_bytes), model_bytes);
    distributor.SetAckMode(
        InitialModelDistributor::AckMode::LoseFirstTransaction);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&distributor);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "initial response-loss distributor starts");
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    MazeServiceImpl service(config);

    Require(MazeServiceUpdateTestAccess::LoadInitialModel(service),
            "initial ACK response loss converges by idempotent retry");
    Require(distributor.ack_calls() == 4 &&
                distributor.last_load_instance_id().find("-load-v1") !=
                    std::string::npos &&
                distributor.last_aiserver().component() == "rl-aiserver" &&
                distributor.last_aiserver().instance_id() != "aiserver-0" &&
                !distributor.last_aiserver().instance_id().empty() &&
                distributor.last_aiserver().lifecycle_epoch() > 0 &&
                distributor.last_load_instance_id().find(
                    distributor.last_aiserver().instance_id()) !=
                    std::string::npos &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) == 1 &&
                MazeServiceUpdateTestAccess::ModelState(service) ==
                    training::MODEL_STATE_READY &&
                MazeServiceUpdateTestAccess::LastError(service).empty() &&
                fs::is_regular_file(
                    root / "cache" / "000001" / "SaveModel.onnx"),
            "initial loss rolls forward locally then receives exact Applied");
    RequireActiveV1(service,
                    "initial response-loss convergence serves tracked v1");

    server->Shutdown();
    server->Wait();
    fs::remove_all(root);

    auto pending_config = Config();
    const fs::path pending_root = fs::temp_directory_path() /
        ("a3-initial-pending-ack-" + std::to_string(getpid()));
    fs::remove_all(pending_root);
    pending_config.model.local_train_dir = pending_root.string();
    pending_config.model.startup_timeout_ms = 250;
    pending_config.model_distribution.rpc_timeout_ms = 25;
    InitialModelDistributor pending_distributor(
        InitialManifest(pending_config, model_bytes), model_bytes);
    pending_distributor.SetAckMode(
        InitialModelDistributor::AckMode::LoseEveryTransaction);
    grpc::ServerBuilder pending_builder;
    int pending_port = 0;
    pending_builder.AddListeningPort("127.0.0.1:0",
                                     grpc::InsecureServerCredentials(),
                                     &pending_port);
    pending_builder.RegisterService(&pending_distributor);
    std::unique_ptr<grpc::Server> pending_server =
        pending_builder.BuildAndStart();
    Require(pending_server != nullptr && pending_port > 0,
            "initial persistent response-loss distributor starts");
    pending_config.model_distribution.host = "127.0.0.1";
    pending_config.model_distribution.port = pending_port;
    MazeServiceImpl pending_service(pending_config);
    MazeServiceUpdateTestAccess::MakeReady(pending_service);

    Require(MazeServiceUpdateTestAccess::LoadInitialModel(pending_service),
            "initial activation rolls forward when exact ACK stays unknown");
    Require(pending_distributor.ack_calls() >= 2 &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(
                    pending_service) == 1 &&
                MazeServiceUpdateTestAccess::AckPending(pending_service) &&
                MazeServiceUpdateTestAccess::ModelState(pending_service) ==
                    training::MODEL_STATE_WAITING &&
                !pending_service.IsReady() &&
                fs::is_regular_file(
                    pending_root / "cache" / "000001" /
                    "SaveModel.onnx"),
            "initial local publication is retained as a pending exact ACK, not "
            "reported as an uncommitted load");
    RequireActiveV1(
        pending_service,
        "initial pending ACK still pins the locally published v1 artifact");

    pending_distributor.SetAckMode(
        InitialModelDistributor::AckMode::Applied);
    Require(MazeServiceUpdateTestAccess::RetryAck(pending_service) &&
                !MazeServiceUpdateTestAccess::AckPending(pending_service) &&
                pending_service.IsReady() &&
                MazeServiceUpdateTestAccess::ModelState(pending_service) ==
                    training::MODEL_STATE_READY,
            "initial exact ACK recovery restores readiness without reloading "
            "the model");

    pending_server->Shutdown();
    pending_server->Wait();
    fs::remove_all(pending_root);
}

void TestLateAgentInferenceFailureHasNoPartialResponse(
    const std::string& failure_fixture) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
        ("a3-update-atomicity-late-infer-" + std::to_string(getpid()));
    fs::remove_all(root);
    fs::create_directories(root / "active");
    const fs::path active = root / "active" / "model.onnx";
    fs::copy_file(failure_fixture, active);

    auto config = Config();
    config.model.local_train_dir = root.string();
    MazeServiceImpl service(config);
    MazeServiceUpdateTestAccess::MakeReady(service);
    MazeServiceUpdateTestAccess::SetModel(service);
    std::string error;
    Require(MazeServiceUpdateTestAccess::LoadActive(
                service, active.string(), error),
            "input-selective failure model loads: " + error);
    auto* session = MazeServiceUpdateTestAccess::AddSession(service);
    Require(session != nullptr, "late-inference session allocated");
    ConfigureSession(*session);
    session->current_max_steps = 8;
    session->agents.at(1).last_action = 1;

    const auto agent_zero_before = session->agents.at(0);
    const auto agent_one_before = session->agents.at(1);
    const std::string rng_before =
        MazeServiceUpdateTestAccess::ActionRng(service);
    auto request = ActiveRequest(*session, 1, {2, 2});
    request.mutable_agents(1)->set_last_move_blocked(true);
    maze::UpdateRsp rejected;
    service.Update(nullptr, &request, &rejected);

    Require(rejected.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_REJECTED &&
                rejected.actions_size() == 0,
            "later Agent inference failure returns no action prefix");
    Require(session->last_frame_id == 0 &&
                session->last_command_sequence == 0 &&
                !session->command_replay.present(),
            "later Agent inference failure commits no frame or replay");
    Require(session->agents.at(0).pending_obs ==
                agent_zero_before.pending_obs &&
                session->agents.at(0).pending_action ==
                    agent_zero_before.pending_action &&
                session->agents.at(1).pending_obs ==
                    agent_one_before.pending_obs &&
                session->agents.at(1).last_move_blocked ==
                    agent_one_before.last_move_blocked &&
                session->agent_sample_caches.empty(),
            "later Agent inference failure preserves both live Agent states");
    Require(MazeServiceUpdateTestAccess::ProducedSamples(service) == 0 &&
                MazeServiceUpdateTestAccess::ProducedBatches(service) == 0 &&
                MazeServiceUpdateTestAccess::FragmentSequence(service) == 0 &&
                MazeServiceUpdateTestAccess::ActionRng(service) == rng_before,
            "later Agent inference failure preserves counters and live RNG");
    fs::remove_all(root);
}

void TestBeginNotAppliedRetryIsAtomic(
    const std::string& active_fixture,
    const std::string& staged_fixture) {
    AuthorityDistributor distributor(Config().contract);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&distributor);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "Begin activation distributor starts");

    const auto paths = InstallDistinctModels(
        "begin-activation", active_fixture, staged_fixture);

    auto config = Config();
    config.model.local_train_dir = paths.root.string();
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    config.model_distribution.rpc_timeout_ms = 50;
    MazeServiceImpl service(config);
    MazeServiceUpdateTestAccess::MakeReady(service);
    MazeServiceUpdateTestAccess::SetModel(service);
    std::string error;
    Require(MazeServiceUpdateTestAccess::LoadActiveAndStage(
                service, paths.active.string(), paths.staged.string(), error),
            "Begin active and staged models load: " + error);
    RequireActiveV0(service, "Begin starts on the distinct v0 model");
    auto* session = MazeServiceUpdateTestAccess::AddSession(service);
    Require(session != nullptr, "Begin activation session allocated");
    ConfigureIdleSession(*session);
    const auto request = BeginRequest(*session, "begin-activation-atomic");
    const std::string controller_before =
        MazeServiceUpdateTestAccess::ControllerJson(service);
    const uint64_t next_episode_before =
        MazeServiceUpdateTestAccess::NextEpisodeId(service);

    distributor.SetStatusAvailable(false);
    maze::BeginEpisodeRsp probe_failed;
    service.BeginEpisode(nullptr, &request, &probe_failed);
    Require(probe_failed.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_WAIT &&
                probe_failed.lifecycle().ret_code() == 0 &&
                probe_failed.lifecycle().error_code() ==
                    maze::LIFECYCLE_ERROR_CODE_UNSPECIFIED &&
                probe_failed.lifecycle().applied_sequence() == 0 &&
                probe_failed.assignment().ByteSizeLong() == 0,
            "pre-write authority outage returns exact-retry WAIT with no "
            "assignment");
    Require(distributor.ack_calls() == 0 &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) == 0 &&
                MazeServiceUpdateTestAccess::StagedModelVersion(service) == 1 &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(service) == 0 &&
                std::filesystem::is_regular_file(paths.active) &&
                std::filesystem::is_regular_file(paths.staged),
            "preflight WAIT sends no write and preserves staged identity");
    Require(session->last_command_sequence == 0 &&
                session->episode_state == SessionManager::EpisodeState::None &&
                MazeServiceUpdateTestAccess::NextEpisodeId(service) ==
                    next_episode_before &&
                MazeServiceUpdateTestAccess::ControllerJson(service) ==
                    controller_before,
            "preflight WAIT preserves Begin session and controller state");
    Require(service.IsReady() &&
                !MazeServiceUpdateTestAccess::SenderSnapshot(service).degraded,
            "pre-write NotApplied keeps the healthy old service retryable");
    RequireActiveV0(service,
                    "probe failure keeps the ordinary serving session on v0");

    distributor.SetStatusAvailable(true);
    distributor.SetMode(AuthorityDistributor::Mode::SameAuthorityApplied);
    maze::BeginEpisodeRsp applied;
    service.BeginEpisode(nullptr, &request, &applied);
    Require(applied.lifecycle().result() == maze::LIFECYCLE_RESULT_APPLIED &&
                applied.assignment().ByteSizeLong() > 0 &&
                session->episode_state == SessionManager::EpisodeState::Active,
            "same Begin command applies after ACK recovery");
    Require(MazeServiceUpdateTestAccess::ActiveModelVersion(service) == 1 &&
                MazeServiceUpdateTestAccess::StagedModelVersion(service) < 0 &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(service) == 1 &&
                MazeServiceUpdateTestAccess::NextEpisodeId(service) ==
                    next_episode_before + 1,
            "successful Begin commits activation and episode exactly once");
    RequireActiveV1(service,
                    "successful Begin switches ordinary serving to distinct v1");

    server->Shutdown();
    server->Wait();
    std::filesystem::remove_all(paths.root);
}

void TestBeginSameAuthorityRejectionRollsBack(
    const std::string& active_fixture,
    const std::string& staged_fixture) {
    AuthorityDistributor distributor(Config().contract);
    distributor.SetMode(
        AuthorityDistributor::Mode::SameAuthorityRejected);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&distributor);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "same-authority rejection distributor starts");

    const auto paths = InstallDistinctModels(
        "begin-rejected", active_fixture, staged_fixture);
    auto config = Config();
    config.model.local_train_dir = paths.root.string();
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    config.model_distribution.rpc_timeout_ms = 50;
    MazeServiceImpl service(config);
    MazeServiceUpdateTestAccess::MakeReady(service);
    MazeServiceUpdateTestAccess::SetModel(service);
    std::string error;
    Require(MazeServiceUpdateTestAccess::LoadActiveAndStage(
                service, paths.active.string(), paths.staged.string(), error),
            "same-reject active and staged models load: " + error);
    auto* session = MazeServiceUpdateTestAccess::AddSession(service);
    Require(session != nullptr, "same-reject Begin session allocated");
    ConfigureIdleSession(*session);
    const auto request = BeginRequest(*session, "begin-same-reject");
    const std::string controller_before =
        MazeServiceUpdateTestAccess::ControllerJson(service);
    const uint64_t next_episode_before =
        MazeServiceUpdateTestAccess::NextEpisodeId(service);

    maze::BeginEpisodeRsp rejected;
    service.BeginEpisode(nullptr, &request, &rejected);
    Require(rejected.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_REJECTED &&
                rejected.assignment().ByteSizeLong() == 0,
            "same-authority rejection publishes no Begin assignment");
    Require(MazeServiceUpdateTestAccess::ActiveModelVersion(service) == 0 &&
                MazeServiceUpdateTestAccess::StagedModelVersion(service) == 1 &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(service) == 0 &&
                MazeServiceUpdateTestAccess::ControllerJson(service) ==
                    controller_before &&
                MazeServiceUpdateTestAccess::NextEpisodeId(service) ==
                    next_episode_before &&
                std::filesystem::is_regular_file(paths.active) &&
                std::filesystem::is_regular_file(paths.staged),
            "same-authority rejection safely rolls back all Begin state");
    RequireActiveV0(service,
                    "same-authority rejection keeps ordinary serving on v0");

    server->Shutdown();
    server->Wait();
    std::filesystem::remove_all(paths.root);
}

void TestBeginAckResponseLossRollsForward(
    const std::string& active_fixture,
    const std::string& staged_fixture) {
    AuthorityDistributor distributor(Config().contract);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&distributor);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "Begin response-loss distributor starts");

    const auto paths = InstallDistinctModels(
        "begin-response-loss", active_fixture, staged_fixture);
    auto config = Config();
    config.model.local_train_dir = paths.root.string();
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    config.model_distribution.rpc_timeout_ms = 50;
    MazeServiceImpl service(config);
    MazeServiceUpdateTestAccess::MakeReady(service);
    MazeServiceUpdateTestAccess::SetModel(service);
    std::string error;
    Require(MazeServiceUpdateTestAccess::LoadActiveAndStage(
                service, paths.active.string(), paths.staged.string(), error),
            "Begin response-loss models load: " + error);
    auto* session = MazeServiceUpdateTestAccess::AddSession(service);
    Require(session != nullptr, "Begin response-loss session allocated");
    ConfigureIdleSession(*session);

    const auto begin_request =
        BeginRequest(*session, "begin-ack-response-loss");
    maze::BeginEpisodeRsp begin_response;
    service.BeginEpisode(nullptr, &begin_request, &begin_response);
    Require(begin_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED &&
                begin_response.assignment().mode() ==
                    maze::EPISODE_MODE_TRAINING &&
                session->episode_state == SessionManager::EpisodeState::Active &&
                session->last_command_sequence == 1 &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) == 1 &&
                MazeServiceUpdateTestAccess::StagedModelVersion(service) < 0 &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(service) == 1 &&
                MazeServiceUpdateTestAccess::AckPending(service),
            "Begin rolls the assignment forward with its locally activated "
            "model after ACK response loss");
    RequireActiveV1(service,
                    "Begin response loss serves the committed v1 model");

    const auto update_request = ActiveRequest(*session, 0, {2, 2});
    maze::UpdateRsp waiting;
    service.Update(nullptr, &update_request, &waiting);
    Require(waiting.lifecycle().result() == maze::LIFECYCLE_RESULT_WAIT &&
                waiting.actions_size() == 0 && session->last_frame_id == -1 &&
                session->last_command_sequence == 1,
            "first Update after Begin activation waits without consuming the "
            "frame");

    distributor.SetMode(AuthorityDistributor::Mode::ChangedAuthorityApplied);
    Require(MazeServiceUpdateTestAccess::RetryAck(service) &&
                !MazeServiceUpdateTestAccess::AckPending(service) &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(service) == 1,
            "Begin activation exact ACK converges without reactivation");
    maze::UpdateRsp applied;
    service.Update(nullptr, &update_request, &applied);
    Require(applied.lifecycle().result() == maze::LIFECYCLE_RESULT_APPLIED &&
                session->last_frame_id == 0 &&
                session->last_command_sequence == 2,
            "same WAITed Update applies after Begin ACK recovery");

    server->Shutdown();
    server->Wait();
    std::filesystem::remove_all(paths.root);
}

void TestWatcherActivationRetriesAfterAuthorityProbe(
    const std::string& active_fixture,
    const std::string& staged_fixture) {
    AuthorityDistributor distributor(Config().contract);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&distributor);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "watcher activation distributor starts");

    const auto paths = InstallDistinctModels(
        "watcher-retry", active_fixture, staged_fixture);
    auto config = Config();
    config.model.local_train_dir = paths.root.string();
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    config.model_distribution.rpc_timeout_ms = 50;
    MazeServiceImpl service(config);
    MazeServiceUpdateTestAccess::MakeReady(service);
    MazeServiceUpdateTestAccess::SetModel(service);
    std::string error;
    Require(MazeServiceUpdateTestAccess::LoadActiveAndStage(
                service, paths.active.string(), paths.staged.string(), error),
            "watcher active and staged models load: " + error);

    const int ack_calls_before_contract = distributor.ack_calls();
    distributor.SetWrongStatusContract(true);
    Require(!MazeServiceUpdateTestAccess::RunWatcherActivationAttempt(service) &&
                distributor.ack_calls() == ack_calls_before_contract &&
                !service.IsReady() &&
                MazeServiceUpdateTestAccess::LastError(service).find(
                    "contract") != std::string::npos &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) == 0 &&
                MazeServiceUpdateTestAccess::StagedModelVersion(service) == 1 &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(service) == 0 &&
                std::filesystem::is_regular_file(paths.active) &&
                std::filesystem::is_regular_file(paths.staged),
            "watcher makes a wrong authority contract terminal before Promote "
            "or ACK");
    RequireActiveV0(service,
                    "wrong watcher status contract keeps serving v0");
    distributor.SetWrongStatusContract(false);
    MazeServiceUpdateTestAccess::MakeReady(service);

    distributor.SetStatusAvailable(false);
    Require(!MazeServiceUpdateTestAccess::RunWatcherActivationAttempt(service) &&
                service.IsReady() &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) == 0 &&
                MazeServiceUpdateTestAccess::StagedModelVersion(service) == 1 &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(service) == 0,
            "watcher defers pre-write failure without degrading old serving");
    RequireActiveV0(service,
                    "deferred watcher activation continues serving v0");

    distributor.SetStatusAvailable(true);
    distributor.SetMode(AuthorityDistributor::Mode::LossAfterApply);
    Require(MazeServiceUpdateTestAccess::RunWatcherActivationAttempt(service) &&
                !service.IsReady() &&
                MazeServiceUpdateTestAccess::AckPending(service) &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) == 1 &&
                MazeServiceUpdateTestAccess::StagedModelVersion(service) < 0 &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(service) == 1,
            "watcher outcome-unknown ACK still commits the staged model once");
    RequireActiveV1(service,
                    "watcher response loss switches ordinary serving to v1");
    auto* pending_session = MazeServiceUpdateTestAccess::AddSession(service);
    Require(pending_session != nullptr,
            "watcher-pending Begin session allocated");
    ConfigureIdleSession(*pending_session);
    const auto pending_begin =
        BeginRequest(*pending_session, "begin-after-watcher-local-activation");
    maze::BeginEpisodeRsp pending_begin_response;
    service.BeginEpisode(
        nullptr, &pending_begin, &pending_begin_response);
    Require(pending_begin_response.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED &&
                pending_begin_response.assignment().behavior_policy()
                        .model_version() == 1 &&
                pending_session->last_command_sequence == 1 &&
                MazeServiceUpdateTestAccess::AckPending(service),
            "Begin binds the complete locally activated watcher model while "
            "its ACK is pending");
    const auto pending_update =
        ActiveRequest(*pending_session, 0, {2, 2});
    maze::UpdateRsp pending_update_wait;
    service.Update(nullptr, &pending_update, &pending_update_wait);
    Require(pending_update_wait.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_WAIT &&
                pending_session->last_frame_id == -1 &&
                pending_session->last_command_sequence == 1,
            "watcher-pending model freezes the first Update frame");

    distributor.SetMode(AuthorityDistributor::Mode::ChangedAuthorityApplied);
    Require(MazeServiceUpdateTestAccess::RetryAck(service) &&
                service.IsReady() &&
                !MazeServiceUpdateTestAccess::AckPending(service) &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(service) == 1,
            "watcher activation converges the same exact ACK without a second "
            "model switch");
    maze::UpdateRsp pending_update_applied;
    service.Update(nullptr, &pending_update, &pending_update_applied);
    Require(pending_update_applied.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED &&
                pending_session->last_frame_id == 0 &&
                pending_session->last_command_sequence == 2,
            "same watcher-pending Update applies after exact ACK recovery");

    const auto fault_paths = InstallDistinctModels(
        "watcher-sender-fault", active_fixture, staged_fixture);
    auto fault_config = Config();
    fault_config.model.local_train_dir = fault_paths.root.string();
    fault_config.model_distribution.host = "127.0.0.1";
    fault_config.model_distribution.port = port;
    fault_config.model_distribution.rpc_timeout_ms = 50;
    MazeServiceImpl fault_service(fault_config);
    MazeServiceUpdateTestAccess::MakeReady(fault_service);
    MazeServiceUpdateTestAccess::SetModel(fault_service);
    Require(MazeServiceUpdateTestAccess::LoadActiveAndStage(
                fault_service, fault_paths.active.string(),
                fault_paths.staged.string(), error),
            "watcher sender-fault models load: " + error);
    const std::string sender_error =
        "sender fault before watcher activation";
    MazeServiceUpdateTestAccess::InjectSenderFault(
        fault_service, sender_error);
    const int ack_calls_before_fault = distributor.ack_calls();
    Require(!MazeServiceUpdateTestAccess::RunWatcherActivationAttempt(
                fault_service) &&
                distributor.ack_calls() == ack_calls_before_fault &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(
                    fault_service) == 0 &&
                MazeServiceUpdateTestAccess::StagedModelVersion(
                    fault_service) == 1 &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(
                    fault_service) == 0 &&
                MazeServiceUpdateTestAccess::LastError(fault_service) ==
                    sender_error,
            "watcher preserves a pre-existing sender fault and skips ACK");
    RequireActiveV0(
        fault_service,
        "sender-degraded watcher attempt keeps ordinary serving on v0");

    server->Shutdown();
    server->Wait();
    std::filesystem::remove_all(paths.root);
    std::filesystem::remove_all(fault_paths.root);
}

void TestBeginPlanFailurePreservesCandidateState() {
    MazeServiceImpl service(Config());
    MazeServiceUpdateTestAccess::MakeReady(service);
    MazeServiceUpdateTestAccess::SetModel(service);
    MazeServiceUpdateTestAccess::SetProducedAndTrained(service, 2);
    MazeServiceUpdateTestAccess::SetManifestDigest(service, "invalid");
    auto* session = MazeServiceUpdateTestAccess::AddSession(service);
    Require(session != nullptr, "Begin plan-failure session allocated");
    ConfigureIdleSession(*session);
    const auto request = BeginRequest(*session, "begin-plan-atomic");
    const std::string controller_before =
        MazeServiceUpdateTestAccess::ControllerJson(service);
    const uint64_t next_episode_before =
        MazeServiceUpdateTestAccess::NextEpisodeId(service);

    maze::BeginEpisodeRsp rejected;
    service.BeginEpisode(nullptr, &request, &rejected);
    Require(rejected.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_REJECTED &&
                rejected.assignment().ByteSizeLong() == 0,
            "post-plan model validation rejects with no partial assignment");
    Require(MazeServiceUpdateTestAccess::ControllerJson(service) ==
                controller_before &&
                MazeServiceUpdateTestAccess::NextEpisodeId(service) ==
                    next_episode_before &&
                session->last_command_sequence == 0 &&
                session->episode_state == SessionManager::EpisodeState::None &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(service) == 0,
            "post-plan validation failure discards candidate controller state");

    MazeServiceUpdateTestAccess::SetManifestDigest(
        service, std::string(64, 'b'));
    maze::BeginEpisodeRsp applied;
    service.BeginEpisode(nullptr, &request, &applied);
    Require(applied.lifecycle().result() == maze::LIFECYCLE_RESULT_APPLIED &&
                applied.assignment().mode() ==
                    maze::EPISODE_MODE_TRAINING &&
                applied.assignment().collect_training_samples() &&
                session->last_command_sequence == 1 &&
                MazeServiceUpdateTestAccess::NextEpisodeId(service) ==
                    next_episode_before + 1,
            "corrected identity commits the planned training episode once");
}

void TestAckResponseLossRollsForward(
    const std::string& active_fixture,
    const std::string& staged_fixture) {
    AuthorityDistributor distributor(Config().contract);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&distributor);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "response-loss distributor starts");

    const auto paths = InstallDistinctModels(
        "response-loss", active_fixture, staged_fixture);
    auto config = Config();
    config.model.local_train_dir = paths.root.string();
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    config.model_distribution.rpc_timeout_ms = 50;
    MazeServiceImpl service(config);
    MazeServiceUpdateTestAccess::MakeReady(service);
    MazeServiceUpdateTestAccess::SetModel(service);
    std::string error;
    Require(MazeServiceUpdateTestAccess::LoadActiveAndStage(
                service, paths.active.string(), paths.staged.string(), error),
            "response-loss active and staged models load: " + error);
    RequireActiveV0(service, "response-loss path starts serving v0");
    auto* session = MazeServiceUpdateTestAccess::AddSession(service);
    Require(session != nullptr, "response-loss session allocated");
    ConfigureSession(*session);
    session->current_max_steps = 8;

    const auto first = ActiveRequest(*session, 1, {2, 2});
    distributor.SetStatusAvailable(false);
    maze::UpdateRsp deferred;
    service.Update(nullptr, &first, &deferred);
    Require(deferred.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_WAIT &&
                deferred.lifecycle().ret_code() == 0 &&
                deferred.lifecycle().error_code() ==
                    maze::LIFECYCLE_ERROR_CODE_UNSPECIFIED &&
                deferred.environment_control() ==
                    maze::ENVIRONMENT_CONTROL_WAIT_FOR_TRAINING_CAPACITY &&
                deferred.retry_after_ms() > 0 &&
                deferred.actions_size() == 0 &&
                session->last_frame_id == 0 &&
                session->last_command_sequence == 0 &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) == 0 &&
                MazeServiceUpdateTestAccess::StagedModelVersion(service) == 1 &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(service) == 0 &&
                !MazeServiceUpdateTestAccess::AckPending(service) &&
                service.IsReady(),
            "Update authority preflight outage is an atomic exact-retry WAIT");
    RequireActiveV0(service,
                    "deferred Update continues ordinary serving on v0");

    distributor.SetStatusAvailable(true);
    maze::UpdateRsp applied;
    service.Update(nullptr, &first, &applied);
    Require(distributor.applied(),
            "remote applied ACK before response loss");
    Require(distributor.last_load_instance_id().find("-load-v1") !=
                std::string::npos,
            "ACK retries preserve deterministic model identity");
    Require(applied.lifecycle().result() == maze::LIFECYCLE_RESULT_APPLIED,
            "outcome-uncertain ACK rolls the prepared frame forward");
    Require(MazeServiceUpdateTestAccess::ActiveModelVersion(service) == 1 &&
                MazeServiceUpdateTestAccess::AckPending(service),
            "local serving identity rolls forward and records pending ACK");
    for (const auto& item : session->agents) {
        Require(item.second.has_pending_action &&
                    item.second.pending_model_version == 1,
                "returned actions are bound to the rolled-forward model");
    }
    Require(!service.IsReady(),
            "new Updates pause while ACK confirmation is pending");
    RequireActiveV1(service,
                    "outcome-uncertain activation serves distinct v1 locally");

    static constexpr int directions[9][2] = {
        {0, 0}, {0, 1}, {1, 1}, {1, 0}, {1, -1},
        {0, -1}, {-1, -1}, {-1, 0}, {-1, 1},
    };
    std::vector<int> next_positions;
    for (const auto& action : applied.actions()) {
        const int candidate = 2 + directions[action.action_id()][0];
        const bool horizontal = directions[action.action_id()][1] == 0;
        next_positions.push_back(
            horizontal && candidate >= 0 && candidate < 3 ? candidate : 2);
    }
    auto second = ActiveRequest(*session, 2, next_positions);
    for (int index = 0; index < second.agents_size(); ++index) {
        second.mutable_agents(index)->set_last_move_blocked(
            applied.actions(index).action_id() != 0 &&
            next_positions[static_cast<std::size_t>(index)] == 2);
    }
    const auto agent_before_wait = session->agents.at(0);
    const int64_t produced_before_wait =
        MazeServiceUpdateTestAccess::ProducedSamples(service);
    maze::UpdateRsp pending_wait;
    service.Update(nullptr, &second, &pending_wait);
    Require(pending_wait.lifecycle().result() == maze::LIFECYCLE_RESULT_WAIT &&
                pending_wait.lifecycle().message().find("model ACK recovery") !=
                    std::string::npos &&
                pending_wait.environment_control() ==
                    maze::ENVIRONMENT_CONTROL_WAIT_FOR_TRAINING_CAPACITY &&
                pending_wait.actions_size() == 0 &&
                session->last_frame_id == 1 &&
                session->last_command_sequence == 1 &&
                session->agents.at(0).pending_obs ==
                    agent_before_wait.pending_obs &&
                MazeServiceUpdateTestAccess::ProducedSamples(service) ==
                    produced_before_wait,
            "next Update waits for exact ACK recovery without consuming the "
            "frame or transition");

    Require(!MazeServiceUpdateTestAccess::RetryAck(service) &&
                MazeServiceUpdateTestAccess::AckPending(service),
            "a second lost response keeps the same ACK pending");
    distributor.SetMode(AuthorityDistributor::Mode::ContradictoryApplied);
    Require(!MazeServiceUpdateTestAccess::RetryAck(service) &&
                MazeServiceUpdateTestAccess::AckPending(service),
            "contradictory ACK code/result remains outcome-uncertain");
    distributor.SetMode(AuthorityDistributor::Mode::InvalidAuthorityApplied);
    Require(!MazeServiceUpdateTestAccess::RetryAck(service) &&
                MazeServiceUpdateTestAccess::AckPending(service),
            "an ACK without a valid authority remains outcome-uncertain");
    distributor.SetMode(AuthorityDistributor::Mode::Unspecified);
    Require(!MazeServiceUpdateTestAccess::RetryAck(service) &&
                MazeServiceUpdateTestAccess::AckPending(service),
            "an unspecified ACK result remains outcome-uncertain");
    distributor.SetMode(
        AuthorityDistributor::Mode::ChangedAuthorityRejected);
    Require(!MazeServiceUpdateTestAccess::RetryAck(service) &&
                MazeServiceUpdateTestAccess::AckPending(service) &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) == 1,
            "new-authority rejection cannot roll back the pinned ACK");
    RequireActiveV1(service,
                    "changed-authority rejection keeps fail-forward serving v1");

    distributor.SetMode(AuthorityDistributor::Mode::ChangedAuthorityApplied);
    Require(MazeServiceUpdateTestAccess::RetryAck(service),
            "a valid positive response converges across authority change");
    Require(service.IsReady() &&
                !MazeServiceUpdateTestAccess::AckPending(service) &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) == 1,
            "ACK recovery preserves local serving identity and readiness");

    maze::UpdateRsp second_applied;
    service.Update(nullptr, &second, &second_applied);
    Require(second_applied.lifecycle().result() ==
                maze::LIFECYCLE_RESULT_APPLIED &&
                session->last_frame_id == 2,
            "next frame applies after pending ACK convergence");
    for (const auto& item : session->agents) {
        const auto& agent = item.second;
        Require(agent.episode_behavior_model_seen &&
                    agent.episode_behavior_model_version_min == 0 &&
                    agent.episode_behavior_model_version_max == 1 &&
                    agent.episode_behavior_model_lineage_id ==
                        "atomicity-fixture",
                "Episode behavior facts span both fragment model versions");
    }

    server->Shutdown();
    server->Wait();
    std::filesystem::remove_all(paths.root);
}

void TestRealWatcherLatestFirstAndPendingAckBackfill(
    const std::string& active_fixture,
    const std::string& distributed_fixture) {
    namespace fs = std::filesystem;
    const std::string model_bytes = ReadBytes(distributed_fixture);
    auto config = Config();
    std::vector<training::ModelArtifactManifest> manifests;
    for (uint64_t version = 0; version <= 2; ++version) {
        manifests.push_back(InitialManifest(
            config, model_bytes, version, static_cast<int64_t>(version)));
    }
    WatcherDistributor distributor(std::move(manifests), model_bytes);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&distributor);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "real watcher distributor starts");

    const fs::path root = fs::temp_directory_path() /
        ("a3-real-watcher-" + std::to_string(getpid()));
    fs::remove_all(root);
    fs::create_directories(root / "active");
    const fs::path active_path = root / "active" / "SaveModel.onnx";
    fs::copy_file(active_fixture, active_path);
    config.model.local_train_dir = root.string();
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    config.model_distribution.rpc_timeout_ms = 50;
    config.model_distribution.poll_interval_ms = 50;

    MazeServiceImpl service(config);
    MazeServiceUpdateTestAccess::MakeReady(service);
    MazeServiceUpdateTestAccess::SetModel(service, 0, 0, 0);
    std::string error;
    Require(MazeServiceUpdateTestAccess::LoadActive(
                service, active_path.string(), error),
            "real watcher active model loads: " + error);
    distributor.SetAckAvailable(false);
    MazeServiceUpdateTestAccess::StartWatcher(service);

    const auto activation_deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(3);
    while ((!MazeServiceUpdateTestAccess::AckPending(service) ||
            MazeServiceUpdateTestAccess::ActiveModelVersion(service) != 2) &&
           std::chrono::steady_clock::now() < activation_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    Require(MazeServiceUpdateTestAccess::AckPending(service) &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) == 2 &&
                MazeServiceUpdateTestAccess::ModelSwitchCount(service) == 1,
            "watcher downloads, prepares, and activates latest before backfill");
    std::this_thread::sleep_for(std::chrono::milliseconds(175));
    Require(distributor.DownloadVersions() == std::vector<uint64_t>{2},
            "pending ACK suppresses every older-version backfill download");

    distributor.SetAckAvailable(true);
    const auto backfill_deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(3);
    while ((MazeServiceUpdateTestAccess::AckPending(service) ||
            !fs::exists(root / "cache" / "000000" / "SaveModel.onnx") ||
            !fs::exists(root / "cache" / "000001" / "SaveModel.onnx")) &&
           std::chrono::steady_clock::now() < backfill_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    MazeServiceUpdateTestAccess::StopWatcher(service);
    const auto downloads = distributor.DownloadVersions();
    Require(!MazeServiceUpdateTestAccess::AckPending(service) &&
                service.IsReady() &&
                downloads == std::vector<uint64_t>({2, 0, 1}) &&
                fs::exists(root / "cache" / "000002" / "SaveModel.onnx"),
            "watcher converges ACK then backfills exactly one oldest gap per cycle");

    server->Shutdown();
    server->Wait();
    fs::remove_all(root);
}

void TestAckRecoveryPreservesSenderFault(
    const std::string& active_fixture,
    const std::string& staged_fixture) {
    AuthorityDistributor distributor(Config().contract);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&distributor);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "sender-fault distributor starts");

    const auto paths = InstallDistinctModels(
        "sender-fault", active_fixture, staged_fixture);
    auto config = Config();
    config.model.local_train_dir = paths.root.string();
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    config.model_distribution.rpc_timeout_ms = 50;
    MazeServiceImpl service(config);
    MazeServiceUpdateTestAccess::MakeReady(service);
    MazeServiceUpdateTestAccess::SetModel(service);
    std::string error;
    Require(MazeServiceUpdateTestAccess::LoadActiveAndStage(
                service, paths.active.string(), paths.staged.string(), error),
            "sender-fault active and staged models load: " + error);
    auto* session = MazeServiceUpdateTestAccess::AddSession(service);
    Require(session != nullptr, "sender-fault session allocated");
    ConfigureSession(*session);
    session->current_max_steps = 8;
    const auto first = ActiveRequest(*session, 1, {2, 2});
    maze::UpdateRsp applied;
    service.Update(nullptr, &first, &applied);
    Require(MazeServiceUpdateTestAccess::AckPending(service),
            "sender-fault setup reaches pending ACK");

    const std::string sender_error = "injected sender transport fault";
    MazeServiceUpdateTestAccess::InjectSenderFault(service, sender_error);
    distributor.SetMode(AuthorityDistributor::Mode::ChangedAuthorityApplied);
    Require(!MazeServiceUpdateTestAccess::RetryAck(service) &&
                !MazeServiceUpdateTestAccess::AckPending(service) &&
                !service.IsReady() &&
                MazeServiceUpdateTestAccess::LastError(service) ==
                    sender_error,
            "ACK success cannot overwrite a concurrent sender fault");
    const auto sender = MazeServiceUpdateTestAccess::SenderSnapshot(service);
    Require(sender.degraded && sender.last_error == sender_error &&
                MazeServiceUpdateTestAccess::ActiveModelVersion(service) == 1,
            "sender degradation and rolled-forward serving identity remain");

    server->Shutdown();
    server->Wait();
    std::filesystem::remove_all(paths.root);
}

void TestShutdownPropagatesUnresolvedPushOutcome() {
    auto config = Config();
    config.sample_output.max_attempts = 1;
    config.sample_output.drain_timeout_ms = 10;
    config.sample_output.status_poll_interval_ms = 10000;

    ShutdownResponseLossDistributor distributor(config.contract);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&distributor);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "shutdown response-loss distributor starts");
    config.sample_output.host = "127.0.0.1";
    config.sample_output.port = port;

    MazeServiceImpl service(config);
    Require(MazeServiceUpdateTestAccess::StartSampleSender(service),
            "service-level shutdown sender starts");
    training::SampleBatch batch;
    batch.set_batch_id("shutdown-unresolved-push");
    batch.mutable_payload_digest()->set_algorithm(
        common::DIGEST_ALGORITHM_SHA256);
    batch.mutable_payload_digest()->set_hex(std::string(64, 'd'));
    batch.set_created_at_unix_ms(UnixNowMs());
    batch.mutable_behavior_policy()->set_model_version(1);
    batch.add_samples();
    batch.add_samples();
    Require(MazeServiceUpdateTestAccess::EnqueueSample(service, batch),
            "service-level shutdown batch enqueues");

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(2);
    while (distributor.push_calls() == 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    Require(distributor.push_calls() > 0,
            "service-level shutdown fault reaches PushSamples");
    Require(!service.BeginShutdown(),
            "unresolved Push outcome reported a clean AIServer shutdown");
    const auto snapshot =
        MazeServiceUpdateTestAccess::SenderSnapshot(service);
    Require(snapshot.unresolved_push_outcome_unknown_samples == 2 &&
                snapshot.unresolved_push_outcome_unknown_batches == 1 &&
                snapshot.final_drop_unique_samples == 0 &&
                snapshot.final_drop_unique_batches == 0 &&
                MazeServiceUpdateTestAccess::ServiceState(service) ==
                    training::AISERVER_STATE_DEGRADED &&
                MazeServiceUpdateTestAccess::LastError(service).find(
                    "unresolved_push_samples=2") != std::string::npos,
            "AIServer shutdown did not retain unresolved/final-drop evidence");
    Require(!service.BeginShutdown(),
            "repeated shutdown erased the latched failure result");

    server->Shutdown();
    server->Wait();
}

void TestShutdownRejectsUnsettledModelAndAccounting() {
    {
        MazeServiceImpl service(Config());
        MazeServiceUpdateTestAccess::MakeReady(service);
        MazeServiceUpdateTestAccess::SetModel(service);
        MazeServiceUpdateTestAccess::SetPendingModelAck(
            service, "injected shutdown ACK response loss");

        Require(!service.BeginShutdown(),
                "pending model ACK reported a clean shutdown");
        Require(MazeServiceUpdateTestAccess::ServiceState(service) ==
                    training::AISERVER_STATE_DEGRADED &&
                    MazeServiceUpdateTestAccess::AckPending(service) &&
                    MazeServiceUpdateTestAccess::LastError(service).find(
                        "model_ack_pending=1") != std::string::npos &&
                    MazeServiceUpdateTestAccess::LastError(service).find(
                        "shutdown ACK response loss") != std::string::npos,
                "shutdown erased the pending model ACK authority evidence");
    }

    {
        MazeServiceImpl service(Config());
        MazeServiceUpdateTestAccess::MakeReady(service);
        MazeServiceUpdateTestAccess::SetModel(service);
        auto* session = MazeServiceUpdateTestAccess::AddSession(service);
        Require(session != nullptr,
                "shutdown accounting session allocated");
        ConfigureSession(*session);
        MazeServiceUpdateTestAccess::AddInconsistentShutdownCache(
            service, *session);

        Require(!service.BeginShutdown(),
                "quarantine accounting corruption reported a clean shutdown");
        Require(MazeServiceUpdateTestAccess::ServiceState(service) ==
                    training::AISERVER_STATE_DEGRADED &&
                    MazeServiceUpdateTestAccess::LastError(service).find(
                        "quarantined behavior-model accounting is inconsistent") !=
                        std::string::npos,
                "shutdown STOPPED state overwrote the quarantine invariant "
                "failure");
    }
}

}  // namespace

int main(int argc, char** argv) {
    Require(argc == 4,
            "usage: update_atomicity_test V0_ONNX V1_ONNX FAILURE_ONNX");
    {
        MazeServiceImpl first(Config());
        MazeServiceImpl second(Config());
        Require(!MazeServiceUpdateTestAccess::ProducerInstanceId(first).empty() &&
                    MazeServiceUpdateTestAccess::ProducerInstanceId(first) !=
                        MazeServiceUpdateTestAccess::ProducerInstanceId(second),
                "same-label producer processes receive distinct instance IDs");
        Require(MazeServiceUpdateTestAccess::ProducerLifecycleEpoch(first) > 0 &&
                    MazeServiceUpdateTestAccess::ProducerLifecycleEpoch(second) > 0,
                "producer lifecycle epochs are non-zero");
    }
    TestMultiAgentUpdateIsAtomic();
    TestStandaloneEvaluationDoesNotRunTrainingReward();
    TestUpdateCapabilityReadinessAndErrors(argv[1]);
    TestBeginAllowsRecoverableSenderStates();
    TestResumeUpdateRejectsCounterRollbackAndAcceptsRepublish(
        argv[1], argv[2]);
    TestLoadInitialAuthorityPreflightIsRetryable(argv[2]);
    TestLoadInitialWrongContractIsRetryable(argv[2]);
    TestLoadInitialResponseLossConverges(argv[2]);
    TestLateAgentInferenceFailureHasNoPartialResponse(argv[3]);
    TestBeginNotAppliedRetryIsAtomic(argv[1], argv[2]);
    TestBeginSameAuthorityRejectionRollsBack(argv[1], argv[2]);
    TestBeginAckResponseLossRollsForward(argv[1], argv[2]);
    TestWatcherActivationRetriesAfterAuthorityProbe(argv[1], argv[2]);
    TestRealWatcherLatestFirstAndPendingAckBackfill(argv[1], argv[2]);
    TestBeginPlanFailurePreservesCandidateState();
    TestAckResponseLossRollsForward(argv[1], argv[2]);
    TestAckRecoveryPreservesSenderFault(argv[1], argv[2]);
    TestShutdownPropagatesUnresolvedPushOutcome();
    TestShutdownRejectsUnsettledModelAndAccounting();
    std::cout << "update_atomicity_contract: PASS\n";
    return 0;
}
