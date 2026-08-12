#include "grpc/maze_service.h"

#include "log/logger.h"
#include "ai/maze_observation.h"
#include "contracts/sample_producer_identity.h"
#include "model/fragment_boundary.h"
#include "model/model_boundary.h"
#include "sample/training_transition_builder.h"
#include "task/maze_map_contract.h"
#include "task/episode_action_policy.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <openssl/evp.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <thread>
#include <unistd.h>

namespace {

double ElapsedMs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

bool IsTerminalReason(maze::MazeTerminationReason reason) {
    return reason == maze::MAZE_TERMINATION_REASON_GOAL_REACHED ||
           reason == maze::MAZE_TERMINATION_REASON_TIME_LIMIT;
}

constexpr int kActionDirections[9][2] = {
    {0, 0}, {0, 1}, {1, 1}, {1, 0}, {1, -1},
    {0, -1}, {-1, -1}, {-1, 0}, {-1, 1},
};

bool ExpectedClientPosition(const SessionManager::Session& session,
                            int from_gx, int from_gy, int action,
                            int& expected_gx, int& expected_gy) {
    if (!session.IsWalkable(from_gx, from_gy) ||
        action < 0 || action >= 9) {
        return false;
    }
    expected_gx = from_gx;
    expected_gy = from_gy;
    const int dx = kActionDirections[action][0];
    const int dy = kActionDirections[action][1];
    const int candidate_gx = from_gx + dx;
    const int candidate_gy = from_gy + dy;
    if (!session.IsWalkable(candidate_gx, candidate_gy)) return true;
    if (dx != 0 && dy != 0 &&
        (!session.IsWalkable(from_gx + dx, from_gy) ||
         !session.IsWalkable(from_gx, from_gy + dy))) {
        return true;
    }
    expected_gx = candidate_gx;
    expected_gy = candidate_gy;
    return true;
}

maze::WorkloadMode WorkloadModeForRunMode(int run_mode) {
    switch (run_mode) {
        case aiserver_mode::kTraining:
            return maze::WORKLOAD_MODE_TRAINING;
        case aiserver_mode::kLocalTest:
            return maze::WORKLOAD_MODE_INFERENCE_SMOKE;
        case aiserver_mode::kModelEvaluation:
            return maze::WORKLOAD_MODE_MODEL_EVALUATION;
        case aiserver_mode::kMapValidation:
            return maze::WORKLOAD_MODE_MAP_VALIDATION;
        default:
            return maze::WORKLOAD_MODE_UNSPECIFIED;
    }
}

maze::ReplayPolicy ReplayPolicyForRunMode(int run_mode) {
    if (run_mode == aiserver_mode::kLocalTest ||
        run_mode == aiserver_mode::kModelEvaluation) {
        return maze::REPLAY_POLICY_RECORD_AND_SERVE;
    }
    return maze::REPLAY_POLICY_DISABLED;
}

constexpr uint32_t kSessionProtocolVersion = 3;
constexpr const char* kObservationSchemaId = "maze.observation.v3";
constexpr const char* kActionSchemaId = "maze.action.v1";
constexpr const char* kActionRuleId =
    "maze.action.9-way.no-corner-cut.v1";

void FillDigest(const DigestConfig& source, common::ContentDigest* target) {
    target->set_algorithm(common::DIGEST_ALGORITHM_SHA256);
    target->set_hex(source.hex);
}

void FillSchema(const SchemaConfig& source, common::SchemaIdentity* target) {
    target->set_schema_id(source.schema_id);
    target->set_schema_version(source.schema_version);
    FillDigest(source.canonical_digest, target->mutable_canonical_digest());
}

void FillContract(const AIServerConfig& config,
                  common::ContractIdentity* target) {
    target->set_package_name(config.contract.package_name);
    target->set_package_version(config.contract.package_version);
    FillDigest(config.contract.source_digest, target->mutable_source_digest());
    FillDigest(config.contract.artifact_digest,
               target->mutable_artifact_digest());
    target->set_platform(config.contract.platform);
    target->set_generator_identity(config.contract.generator_identity);
}

void FillTrainingSemantics(
    const AIServerConfig& config,
    training::TrainingSemanticsIdentity* target) {
    target->set_training_contract_id(
        config.training_semantics.training_contract_id);
    FillSchema(config.training_semantics.observation_schema,
               target->mutable_observation_schema());
    FillSchema(config.training_semantics.action_schema,
               target->mutable_action_schema());
    FillSchema(config.training_semantics.reward_schema,
               target->mutable_reward_schema());
    target->set_policy_distribution_schema_id(
        config.training_semantics.policy_distribution_schema_id);
    target->set_model_architecture_id(
        config.training_semantics.model_architecture_id);
    FillDigest(config.training_semantics.semantics_digest,
               target->mutable_semantics_digest());
}

void FillTaskIdentity(const AIServerConfig& config,
                      maze::TaskIdentity* target) {
    target->set_task_contract_id(config.task.task_contract_id);
    target->set_task_id(config.task.task_id);
    target->set_task_revision(config.task.task_revision);
    FillDigest(config.task.task_config_digest,
               target->mutable_task_config_digest());
    target->set_fixed_map_id(config.task.fixed_map_id);
    target->mutable_fixed_map_digest()->set_algorithm(
        common::DIGEST_ALGORITHM_SHA256);
    target->mutable_fixed_map_digest()->set_hex(
        config.task.fixed_map_checksum_sha256);
}

void FillBehaviorPolicy(const AIServerConfig& config,
                        const ModelManifest& manifest,
                        maze::BehaviorPolicyBinding* target) {
    target->set_model_lineage_id(manifest.model_lineage_id);
    target->set_model_version(
        static_cast<uint64_t>(manifest.model_version));
    target->mutable_model_artifact_digest()->set_algorithm(
        common::DIGEST_ALGORITHM_SHA256);
    target->mutable_model_artifact_digest()->set_hex(manifest.sha256);
    target->mutable_model_manifest_digest()->set_algorithm(
        common::DIGEST_ALGORITHM_SHA256);
    target->mutable_model_manifest_digest()->set_hex(
        manifest.manifest_digest);
    target->set_distribution_schema_id(
        config.policy.distribution_schema_id);
    FillDigest(config.policy.policy_spec_digest,
               target->mutable_policy_spec_digest());
}

void FillServiceIdentity(const std::string& component,
                         const std::string& instance_id,
                         uint64_t lifecycle_epoch,
                         common::ServiceInstanceIdentity* target) {
    target->set_component(component);
    target->set_instance_id(instance_id);
    target->set_lifecycle_epoch(lifecycle_epoch);
}

std::string DeterministicBytes(const google::protobuf::MessageLite& message) {
    std::string output;
    output.resize(message.ByteSizeLong());
    google::protobuf::io::ArrayOutputStream array(
        output.data(), static_cast<int>(output.size()));
    google::protobuf::io::CodedOutputStream coded(&array);
    coded.SetSerializationDeterministic(true);
    if (!message.SerializeToCodedStream(&coded) || coded.HadError()) {
        return "";
    }
    output.resize(static_cast<std::size_t>(coded.ByteCount()));
    return output;
}

std::string Sha256Bytes(const std::string& payload) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context) return "";
    bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
              EVP_DigestUpdate(context, payload.data(), payload.size()) == 1;
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int size = 0;
    if (ok) ok = EVP_DigestFinal_ex(context, digest.data(), &size) == 1;
    EVP_MD_CTX_free(context);
    if (!ok) return "";
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < size; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    return output.str();
}

bool SameSchema(const common::SchemaIdentity& value,
                const SchemaConfig& expected) {
    common::SchemaIdentity identity;
    FillSchema(expected, &identity);
    return value.SerializeAsString() == identity.SerializeAsString();
}

bool ContainsSchema(
    const google::protobuf::RepeatedPtrField<common::SchemaIdentity>& values,
    const SchemaConfig& expected) {
    return std::any_of(values.begin(), values.end(), [&](const auto& value) {
        return SameSchema(value, expected);
    });
}

}  // namespace

MazeServiceImpl::MazeServiceImpl(const AIServerConfig& config)
    : config_(config),
      sample_sender_(config),
      episode_metrics_(config.metrics.episode_window),
      model_distributor_(config),
      producer_instance_id_(
          CreateProducerInstanceId(config.sample_output.aiserver_id)),
      next_lifecycle_epoch_(
          static_cast<uint64_t>(NowMs() / 1000) * 100000ULL +
          static_cast<uint64_t>(getpid() % 100000)),
      action_rng_(config.policy.sampling_seed),
      current_fragment_samples_(config.sample_output.fragment_samples) {}

MazeServiceImpl::~MazeServiceImpl() {
    BeginShutdown();
}

int64_t MazeServiceImpl::NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

SingleMapModelIdentity MazeServiceImpl::ActiveModelIdentity() const {
    SingleMapModelIdentity identity;
    identity.model_version = model_manifest_.model_version;
    identity.model_checksum = model_manifest_.sha256;
    identity.train_updates = model_manifest_.train_updates;
    identity.trained_samples = model_manifest_.trained_samples;
    return identity;
}

bool MazeServiceImpl::ValidateStagedModelProgress(
    const ModelManifest& active,
    const ModelManifest& candidate,
    std::string& error) const {
    if (active.model_version < 0 || active.train_updates < 0 ||
        active.trained_samples < 0 ||
        candidate.model_version <= active.model_version ||
        candidate.train_updates < active.train_updates ||
        candidate.trained_samples < active.trained_samples) {
        error = "staged publication must advance without training-counter rollback";
        return false;
    }
    return true;
}

bool MazeServiceImpl::WriteTaskControllerReceipt(
    const SingleMapTaskController& controller,
    std::string& error) const {
    if (task_controller_receipt_writer_) {
        return task_controller_receipt_writer_(controller, error);
    }
    namespace fs = std::filesystem;
    const fs::path root(config_.model.local_train_dir);
    std::error_code filesystem_error;
    fs::create_directories(root, filesystem_error);
    if (filesystem_error) {
        error = "cannot create task receipt directory: " +
                filesystem_error.message();
        return false;
    }
    const fs::path destination = root / "single-map-task-result.json";
    const fs::path temporary =
        destination.string() + ".tmp." + std::to_string(getpid());
    {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output) {
            error = "cannot open temporary task receipt";
            return false;
        }
        output << controller.ToJson() << '\n';
        output.flush();
        if (!output) {
            fs::remove(temporary, filesystem_error);
            error = "cannot flush temporary task receipt";
            return false;
        }
    }
    fs::rename(temporary, destination, filesystem_error);
    if (filesystem_error) {
        fs::remove(temporary, filesystem_error);
        error = "cannot publish task receipt: " +
                filesystem_error.message();
        return false;
    }
    return true;
}

std::string MazeServiceImpl::CreateProducerInstanceId(
    const std::string& aiserver_id) {
    std::ostringstream output;
    output << aiserver_id << "-" << NowMs() << "-" << getpid();
    return output.str();
}

bool MazeServiceImpl::LoadInitialModel() {
    model_state_.store(training::MODEL_STATE_WAITING);

    if (config_.server.run_mode == aiserver_mode::kModelEvaluation) {
        std::string error;
        if (!LoadModelManifestFile(
                config_,
                LocalEvaluationManifestPath(config_.model),
                model_manifest_, error)) {
            model_state_.store(training::MODEL_STATE_FAILED);
            last_error_ = "model-evaluation model manifest failed: " + error;
            return false;
        }
        if (!onnx_inferencer_.LoadModel(
                model_manifest_.model_path,
                config_.model.expected_obs_dim,
                config_.model.expected_action_dim, &error)) {
            model_state_.store(training::MODEL_STATE_FAILED);
            last_error_ = "model-evaluation model load failed: " + error;
            return false;
        }
        model_state_.store(training::MODEL_STATE_READY);
        last_error_.clear();
        return true;
    }

    if (config_.server.run_mode == aiserver_mode::kLocalTest) {
        const std::string manifest_path =
            config_.model.local_test_dir + "/" +
            config_.model.manifest_name;
        std::string error;
        if (!LoadModelManifestFile(
                config_, manifest_path, model_manifest_, error)) {
            model_state_.store(training::MODEL_STATE_FAILED);
            last_error_ = "local-test model manifest failed: " + error;
            return false;
        }
        if (!onnx_inferencer_.LoadModel(
                model_manifest_.model_path,
                config_.model.expected_obs_dim,
                config_.model.expected_action_dim, &error)) {
            model_state_.store(training::MODEL_STATE_FAILED);
            last_error_ = "local-test model validation failed: " + error;
            return false;
        }
        model_state_.store(training::MODEL_STATE_READY);
        last_error_.clear();
        return true;
    }

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(config_.model.startup_timeout_ms);
    std::string error;
    while (std::chrono::steady_clock::now() < deadline) {
        ModelManifest candidate;
        if (model_distributor_.FetchLatest(
                config_.sample_output.aiserver_id,
                candidate, error)) {
            std::string load_error;
            OnnxInferencer::PreparedModel prepared;
            if (!onnx_inferencer_.PrepareModel(
                    candidate.model_path, config_.model.expected_obs_dim,
                    config_.model.expected_action_dim, prepared,
                    &load_error)) {
                std::string ack_error;
                model_distributor_.Ack(
                    candidate, config_.sample_output.aiserver_id,
                    training::MODEL_LOAD_STATUS_FAILED, load_error, ack_error);
                model_state_.store(training::MODEL_STATE_FAILED);
                last_error_ = "ONNX model validation failed: " + load_error;
                return false;
            }
            common::ServiceInstanceIdentity ack_authority;
            while (std::chrono::steady_clock::now() < deadline &&
                   !model_distributor_.ProbeAckAuthority(
                       ack_authority, load_error)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (ack_authority.instance_id().empty()) {
                error = load_error;
                continue;
            }
            const std::string incoming_path = candidate.model_path;
            std::string previous_path;
            if (!model_distributor_.Promote(
                    candidate, previous_path, load_error)) {
                model_state_.store(training::MODEL_STATE_FAILED);
                last_error_ = "model promotion failed: " + load_error;
                return false;
            }
            std::string ack_error;
            auto ack = model_distributor_.AckIdempotently(
                candidate, config_.sample_output.aiserver_id,
                training::MODEL_LOAD_STATUS_LOADED, "loaded", ack_error,
                &ack_authority);
            if (ack == ModelDistributorClient::AckDisposition::Rejected ||
                ack == ModelDistributorClient::AckDisposition::NotApplied) {
                std::string rollback_error;
                model_distributor_.RollbackPromotion(
                    candidate, incoming_path, previous_path, rollback_error);
                model_state_.store(training::MODEL_STATE_FAILED);
                last_error_ = "initial model ACK was rejected: " + ack_error;
                if (!rollback_error.empty()) {
                    last_error_ += "; promotion rollback failed: " +
                                   rollback_error;
                }
                return false;
            }

            while (ack == ModelDistributorClient::AckDisposition::Uncertain &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                ack = model_distributor_.AckIdempotently(
                    candidate, config_.sample_output.aiserver_id,
                    training::MODEL_LOAD_STATUS_LOADED, "loaded", ack_error,
                    &ack_authority);
            }
            if (ack == ModelDistributorClient::AckDisposition::Rejected ||
                ack == ModelDistributorClient::AckDisposition::NotApplied) {
                std::string rollback_error;
                model_distributor_.RollbackPromotion(
                    candidate, incoming_path, previous_path, rollback_error);
                model_state_.store(training::MODEL_STATE_FAILED);
                last_error_ = "initial model ACK was rejected: " + ack_error;
                if (!rollback_error.empty()) {
                    last_error_ += "; promotion rollback failed: " +
                                   rollback_error;
                }
                return false;
            }

            // Local model publication is the point of no return. If the exact
            // ACK remains outcome-unknown, publish once and let the watcher
            // reconcile it; startup must not pretend the activation failed.
            prepared.model_path = candidate.model_path;
            onnx_inferencer_.ActivatePreparedModel(std::move(prepared));
            model_manifest_ = candidate;
            if (ack == ModelDistributorClient::AckDisposition::Uncertain) {
                RecordPendingModelAck(candidate, ack_authority, ack_error);
                return true;
            }
            model_state_.store(training::MODEL_STATE_READY);
            last_error_.clear();
            LOG_INFO("MazeService",
                     "初始模型就绪: version=%d sha256=%s",
                     model_manifest_.model_version,
                     model_manifest_.sha256.c_str());
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    model_state_.store(training::MODEL_STATE_FAILED);
    last_error_ = error.empty() ? "initial model startup timeout" : error;
    return false;
}

void MazeServiceImpl::StartModelWatcher() {
    if (config_.server.run_mode != aiserver_mode::kTraining ||
        model_watch_thread_.joinable()) {
        return;
    }
    model_watch_stop_.store(false);
    model_watch_thread_ =
        std::thread(&MazeServiceImpl::ModelWatchLoop, this);
}

void MazeServiceImpl::StopModelWatcher() {
    model_watch_stop_.store(true);
    if (model_watch_thread_.joinable()) {
        model_watch_thread_.join();
    }
}

void MazeServiceImpl::RecordPendingModelAck(
    const ModelManifest& manifest,
    const common::ServiceInstanceIdentity& authority,
    const std::string& error) {
    model_ack_pending_ = true;
    pending_model_ack_manifest_ = manifest;
    pending_model_ack_authority_ = authority;
    pending_model_ack_error_ = error;
    pending_model_ack_cause_ =
        "model ACK remains outcome-uncertain after local roll-forward: " +
        error;
    model_state_.store(training::MODEL_STATE_WAITING);
    state_.store(training::AISERVER_STATE_DEGRADED);
    last_error_ = pending_model_ack_cause_;
    LOG_ERROR("MazeService", "%s", last_error_.c_str());
}

bool MazeServiceImpl::HasLocallyActivatedPendingModel() const {
    return model_ack_pending_ && onnx_inferencer_.IsLoaded() &&
           model_manifest_.model_version ==
               pending_model_ack_manifest_.model_version &&
           model_manifest_.sha256 == pending_model_ack_manifest_.sha256 &&
           model_manifest_.model_lineage_id ==
               pending_model_ack_manifest_.model_lineage_id &&
           model_manifest_.manifest_digest ==
               pending_model_ack_manifest_.manifest_digest &&
           model_manifest_.train_updates ==
               pending_model_ack_manifest_.train_updates &&
           model_manifest_.trained_samples ==
               pending_model_ack_manifest_.trained_samples &&
           model_manifest_.model_path ==
               pending_model_ack_manifest_.model_path &&
           onnx_inferencer_.GetModelPath() == model_manifest_.model_path;
}

bool MazeServiceImpl::RetryPendingModelAck() {
    ModelManifest pending;
    common::ServiceInstanceIdentity authority;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!model_ack_pending_) return true;
        pending = pending_model_ack_manifest_;
        authority = pending_model_ack_authority_;
    }

    std::string ack_error;
    const auto ack = model_distributor_.AckIdempotently(
        pending, config_.sample_output.aiserver_id,
        training::MODEL_LOAD_STATUS_LOADED, "loaded", ack_error,
        &authority);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!model_ack_pending_ ||
        pending_model_ack_manifest_.model_version != pending.model_version ||
        pending_model_ack_manifest_.sha256 != pending.sha256) {
        return !model_ack_pending_;
    }
    if (ack == ModelDistributorClient::AckDisposition::Applied) {
        const std::string ack_cause = pending_model_ack_cause_;
        model_ack_pending_ = false;
        pending_model_ack_manifest_ = ModelManifest{};
        pending_model_ack_authority_.Clear();
        pending_model_ack_error_.clear();
        pending_model_ack_cause_.clear();
        model_state_.store(training::MODEL_STATE_READY);
        const bool ack_was_only_fault =
            state_.load() == training::AISERVER_STATE_DEGRADED &&
            last_error_ == ack_cause && sample_sender_.IsReady() &&
            !sample_sender_.IsDegraded();
        if (ack_was_only_fault) {
            state_.store(training::AISERVER_STATE_READY);
            last_error_.clear();
        }
        return state_.load() == training::AISERVER_STATE_READY;
    }
    if (ack == ModelDistributorClient::AckDisposition::Rejected ||
        ack == ModelDistributorClient::AckDisposition::NotApplied) {
        model_ack_pending_ = false;
        pending_model_ack_error_ = ack_error;
        model_state_.store(training::MODEL_STATE_FAILED);
        state_.store(training::AISERVER_STATE_DEGRADED);
        last_error_ = "pending model ACK was explicitly rejected: " +
                      ack_error;
        return false;
    }
    const std::string prior_ack_cause = pending_model_ack_cause_;
    pending_model_ack_error_ = ack_error;
    pending_model_ack_cause_ =
        "pending model ACK remains outcome-uncertain: " + ack_error;
    if (last_error_ == prior_ack_cause) {
        last_error_ = pending_model_ack_cause_;
    }
    return false;
}

bool MazeServiceImpl::TryActivateStagedModelForWatcher() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!IsReady() || model_ack_pending_ || !CanActivateStagedModel()) {
        return false;
    }
    return ActivateStagedModel();
}

void MazeServiceImpl::ModelWatchLoop() {
    const auto poll_interval = std::chrono::milliseconds(
        std::max(50, config_.model_distribution.poll_interval_ms));
    while (!model_watch_stop_.load()) {
        bool has_pending_ack = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            has_pending_ack = model_ack_pending_;
        }
        if (has_pending_ack) {
            RetryPendingModelAck();
            auto remaining = poll_interval;
            while (!model_watch_stop_.load() && remaining.count() > 0) {
                const auto slice =
                    std::min(remaining, std::chrono::milliseconds(50));
                std::this_thread::sleep_for(slice);
                remaining -= slice;
            }
            continue;
        }

        int active_version = -1;
        int staged_version = -1;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_version = model_manifest_.model_version;
            staged_version = staged_model_manifest_.model_version;
        }

        int latest_version = -1;
        std::string latest_checksum;
        std::string error;
        if (model_distributor_.GetLatestIdentity(
                config_.sample_output.aiserver_id,
                latest_version, latest_checksum, error) &&
            ShouldFetchModelCandidate(
                active_version, staged_version, latest_version)) {
            const int requested_version = latest_version;
            ModelManifest candidate;
            if (model_distributor_.FetchVersion(
                    config_.sample_output.aiserver_id,
                    requested_version, candidate, error)) {
                OnnxInferencer validator;
                std::string validation_error;
                if (!validator.LoadModel(
                        candidate.model_path,
                        config_.model.expected_obs_dim,
                        config_.model.expected_action_dim,
                        &validation_error)) {
                    std::string ack_error;
                    model_distributor_.Ack(
                        candidate, config_.sample_output.aiserver_id,
                        training::MODEL_LOAD_STATUS_FAILED,
                        validation_error, ack_error);
                    std::lock_guard<std::mutex> lock(mutex_);
                    last_error_ =
                        "staged model validation failed: " +
                        validation_error;
                } else {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (ShouldFetchModelCandidate(
                            model_manifest_.model_version,
                            staged_model_manifest_.model_version,
                            candidate.model_version)) {
                        staged_model_manifest_ = std::move(candidate);
                        LOG_INFO(
                            "MazeService",
                            "模型已暂存: version=%d sha256=%s",
                            staged_model_manifest_.model_version,
                            staged_model_manifest_.sha256.c_str());
                    } else {
                        std::error_code remove_error;
                        std::filesystem::remove(
                            candidate.model_path, remove_error);
                    }
                }
            }
        }
        TryActivateStagedModelForWatcher();

        auto remaining = poll_interval;
        while (!model_watch_stop_.load() &&
               remaining.count() > 0) {
            const auto slice =
                std::min(remaining, std::chrono::milliseconds(50));
            std::this_thread::sleep_for(slice);
            remaining -= slice;
        }
    }
}

bool MazeServiceImpl::Start() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (started_) return state_.load() == training::AISERVER_STATE_READY;

        state_.store(training::AISERVER_STATE_STARTING);
        if (config_.server.run_mode == aiserver_mode::kTraining) {
            if (!sample_sender_.Start()) {
                auto sender = sample_sender_.GetSnapshot();
                last_error_ = sender.last_error;
                state_.store(training::AISERVER_STATE_DEGRADED);
                LOG_ERROR("MazeService", "样本链路启动失败: %s",
                          last_error_.c_str());
                return false;
            }
            if (!LoadInitialModel()) {
                sample_sender_.StopAndDrain();
                state_.store(training::AISERVER_STATE_DEGRADED);
                LOG_ERROR("MazeService", "训练模型加载失败: %s",
                          last_error_.c_str());
                return false;
            }
        } else if (config_.server.run_mode ==
                       aiserver_mode::kModelEvaluation ||
                   config_.server.run_mode == aiserver_mode::kLocalTest) {
            if (!LoadInitialModel()) {
                state_.store(training::AISERVER_STATE_DEGRADED);
                LOG_ERROR("MazeService", "启动模型加载失败: %s",
                          last_error_.c_str());
                return false;
            }
        }

        started_ = true;
        if (!model_ack_pending_) {
            state_.store(training::AISERVER_STATE_READY);
        }
    }
    StartModelWatcher();
    return true;
}

bool MazeServiceImpl::IsReady() const {
    return IsCoreInferenceReady() &&
           (config_.server.run_mode != aiserver_mode::kTraining ||
            sample_sender_.IsTrainingDeliveryReady());
}

bool MazeServiceImpl::IsCoreInferenceReady() const {
    return state_.load() == training::AISERVER_STATE_READY &&
           (config_.server.run_mode == aiserver_mode::kMapValidation ||
            model_state_.load() == training::MODEL_STATE_READY);
}

bool MazeServiceImpl::BeginShutdown() {
    StopModelWatcher();
    bool preexisting_service_fault = false;
    std::string preexisting_service_error;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_started_) {
            return shutdown_completed_ && shutdown_succeeded_;
        }
        preexisting_service_fault =
            state_.load() == training::AISERVER_STATE_DEGRADED;
        preexisting_service_error = last_error_;
        shutdown_started_ = true;
        shutdown_completed_ = false;
        shutdown_succeeded_ = false;
        state_.store(training::AISERVER_STATE_DRAINING);

        for (const auto& session_id : session_mgr_.GetSessionIds()) {
            SessionManager::Session* session =
                session_mgr_.GetSession(session_id);
            if (!session) continue;
            for (auto& item : session->agents) {
                int agent_id = item.first;
                QuarantineAgentSamples(*session, agent_id);
            }
        }
    }

    bool sender_drained = true;
    if (config_.server.run_mode == aiserver_mode::kTraining) {
        sender_drained = sample_sender_.StopAndDrain();
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const int64_t remaining_samples = CountCachedSamples();
        const int64_t remaining_batches = CountCachedFragments();
        if (remaining_samples > 0 || remaining_batches > 0) {
            sample_sender_.RecordFinalDrop(
                remaining_samples, remaining_batches,
                "session cache remained after drain deadline");
        }
        for (const auto& session_id : session_mgr_.GetSessionIds()) {
            SessionManager::Session* session =
                session_mgr_.GetSession(session_id);
            if (!session) continue;
            session->agent_sample_caches.clear();
            session->pending_sample_batches.clear();
        }
        const auto sender = sample_sender_.GetSnapshot();
        const bool shutdown_accounting_fault =
            state_.load() == training::AISERVER_STATE_DEGRADED;
        const std::string shutdown_accounting_error = last_error_;
        const bool model_state_settled =
            !model_ack_pending_ &&
            model_state_.load() != training::MODEL_STATE_FAILED &&
            (!started_ ||
             config_.server.run_mode == aiserver_mode::kMapValidation ||
             model_state_.load() == training::MODEL_STATE_READY);
        shutdown_succeeded_ =
            !preexisting_service_fault && !shutdown_accounting_fault &&
            model_state_settled && sender_drained && !sender.degraded &&
            !sender.terminal_fault &&
            sender.queue_fragments == 0 &&
            sender.unresolved_push_outcome_unknown_samples == 0 &&
            sender.unresolved_push_outcome_unknown_batches == 0 &&
            sender.final_drop_unique_samples == 0 &&
            sender.final_drop_unique_batches == 0;
        shutdown_completed_ = true;
        if (shutdown_succeeded_) {
            state_.store(training::AISERVER_STATE_STOPPED);
        } else {
            std::ostringstream error;
            error << "AIServer shutdown sample disposition failed: drained="
                  << (sender_drained ? 1 : 0)
                  << " outbound_samples=" << sender.queue_samples
                  << " outbound_batches=" << sender.queue_fragments
                  << " unresolved_push_samples="
                  << sender.unresolved_push_outcome_unknown_samples
                  << " unresolved_push_batches="
                  << sender.unresolved_push_outcome_unknown_batches
                  << " final_drop_samples="
                  << sender.final_drop_unique_samples
                  << " final_drop_batches="
                  << sender.final_drop_unique_batches
                  << " model_ack_pending=" << (model_ack_pending_ ? 1 : 0)
                  << " model_state="
                  << static_cast<int>(model_state_.load());
            const std::string& preserved_error =
                !shutdown_accounting_error.empty()
                    ? shutdown_accounting_error
                    : preexisting_service_error;
            if (!preserved_error.empty()) {
                error << " cause=" << preserved_error;
            }
            last_error_ = error.str();
            state_.store(training::AISERVER_STATE_DEGRADED);
            LOG_ERROR("MazeService", "%s", last_error_.c_str());
        }
        return shutdown_succeeded_;
    }
}

bool MazeServiceImpl::AtLocalFragmentBoundary(
    const SessionManager::Session* candidate_session) {
    std::vector<FragmentBoundaryState> agents;
    for (const auto& session_id : session_mgr_.GetSessionIds()) {
        const SessionManager::Session* session =
            candidate_session &&
                    candidate_session->session_id == session_id
                ? candidate_session
                : session_mgr_.GetSession(session_id);
        if (!session ||
            session->episode_state != SessionManager::EpisodeState::Active) {
            continue;
        }
        for (const auto& item : session->agents) {
            const int agent_id = item.first;
            const auto& agent = item.second;
            const auto cache = session->agent_sample_caches.find(agent_id);
            agents.push_back(FragmentBoundaryState{
                agent.has_pending_action,
                cache != session->agent_sample_caches.end() &&
                    !cache->second.empty(),
                session->pending_sample_batches.find(agent_id) !=
                    session->pending_sample_batches.end(),
            });
        }
    }
    return AllLocalAgentsAtFragmentBoundary(agents);
}

bool MazeServiceImpl::ActiveEpisodesAllowModelActivation() {
    std::vector<ActiveBehaviorPolicyState> episodes;
    for (const auto& session_id : session_mgr_.GetSessionIds()) {
        const SessionManager::Session* session =
            session_mgr_.GetSession(session_id);
        if (!session) continue;
        episodes.push_back(ActiveBehaviorPolicyState{
            session->episode_state == SessionManager::EpisodeState::Active,
            session->behavior_policy_scope,
        });
    }
    return ActiveEpisodesAllowFragmentPolicySwitch(episodes);
}

bool MazeServiceImpl::CanActivateStagedModel(
    const SessionManager::Session* candidate_session) {
    return staged_model_manifest_.model_version >= 0 &&
           ActiveEpisodesAllowModelActivation() &&
           AtLocalFragmentBoundary(candidate_session);
}

bool MazeServiceImpl::ActivateStagedModel() {
    if (!CanActivateStagedModel()) return true;

    std::string error;
    if (!ValidateStagedModelProgress(
            model_manifest_, staged_model_manifest_, error)) {
        MarkDegraded("staged model progress is invalid: " + error);
        return false;
    }

    OnnxInferencer::PreparedModel prepared;
    if (!onnx_inferencer_.PrepareModel(
            staged_model_manifest_.model_path,
            config_.model.expected_obs_dim,
            config_.model.expected_action_dim, prepared, &error)) {
        MarkDegraded("staged model preparation failed: " + error);
        return false;
    }
    common::ServiceInstanceIdentity ack_authority;
    const auto authority_probe =
        model_distributor_.ProbeAckAuthorityDisposition(
            ack_authority, error);
    if (authority_probe ==
        ModelDistributorClient::AuthorityProbeDisposition::Retryable) {
        LOG_ERROR("MazeService", "模型激活延后: %s", error.c_str());
        return false;
    }
    if (authority_probe ==
        ModelDistributorClient::AuthorityProbeDisposition::Rejected) {
        MarkDegraded("model ACK authority probe was rejected: " + error);
        return false;
    }
    ModelManifest candidate = staged_model_manifest_;
    const std::string incoming_path = candidate.model_path;
    std::string previous_path;
    if (!model_distributor_.Promote(
            candidate, previous_path, error)) {
        MarkDegraded("staged model promotion failed: " + error);
        return false;
    }

    std::string ack_error;
    const auto ack = model_distributor_.AckIdempotently(
        candidate, config_.sample_output.aiserver_id,
        training::MODEL_LOAD_STATUS_LOADED, "loaded", ack_error,
        &ack_authority);
    if (ack == ModelDistributorClient::AckDisposition::Rejected ||
        ack == ModelDistributorClient::AckDisposition::NotApplied) {
        std::string rollback_error;
        if (!model_distributor_.RollbackPromotion(
                candidate, incoming_path, previous_path, rollback_error)) {
            ack_error += "; promotion rollback failed: " + rollback_error;
        }
        MarkDegraded("model ACK was rejected: " + ack_error);
        return false;
    }

    prepared.model_path = candidate.model_path;
    onnx_inferencer_.ActivatePreparedModel(std::move(prepared));
    model_manifest_ = candidate;
    staged_model_manifest_ = ModelManifest{};
    ++model_switch_count_;
    if (ack == ModelDistributorClient::AckDisposition::Uncertain) {
        RecordPendingModelAck(candidate, ack_authority, ack_error);
        return true;
    }
    model_state_.store(training::MODEL_STATE_READY);
    LOG_INFO(
        "MazeService", "模型切换完成: version=%d sha256=%s",
        model_manifest_.model_version, model_manifest_.sha256.c_str());
    return true;
}

void MazeServiceImpl::RefreshFragmentSampleTarget(
    const SessionManager::Session& session) {
    bool all_agents_active = !session.agents.empty();
    for (const auto& item : session.agents) {
        if (item.second.done_collected) {
            all_agents_active = false;
            break;
        }
    }
    current_fragment_samples_ = SelectPerAgentFragmentSamples(
        produced_unique_samples_,
        static_cast<int64_t>(config_.task.agent_num) *
            static_cast<int64_t>(config_.sample_output.fragment_samples),
        config_.task.agent_num,
        config_.sample_output.fragment_samples,
        all_agents_active);
}

void MazeServiceImpl::InitAgentSolver(
    SessionManager::AgentRuntime& agent,
    const SessionManager::Session& session) {
    agent.path_valid = agent.solver.InitGrid(
        session.grid_cols, session.grid_rows, session.blocked);
    if (agent.path_valid) {
        agent.path_valid = agent.solver.PlanPath(
            session.start_gx, session.start_gy,
            session.end_gx, session.end_gy);
    }
}

void MazeServiceImpl::ResetEpisodeState(SessionManager::Session& session,
                                        const std::string& episode_id) {
    session.current_episode_id = episode_id;
    session.episode_state = SessionManager::EpisodeState::Active;
    session.last_frame_id = -1;
    session.last_actions.clear();
    session.agent_sample_caches.clear();
    session.pending_sample_batches.clear();

    for (auto& item : session.agents) {
        auto& agent = item.second;
        agent.prev_grid_x = -1;
        agent.prev_grid_y = -1;
        agent.reached_goal = false;
        agent.done_collected = false;
        agent.has_pending_action = false;
        agent.pending_action = 0;
        agent.pending_action_frame_id = -1;
        agent.pending_log_prob = 0.0f;
        agent.pending_value = 0.0f;
        agent.pending_model_version = -1;
        agent.pending_model_checksum.clear();
        agent.pending_model_lineage_id.clear();
        agent.pending_model_manifest_digest.clear();
        agent.pending_obs.clear();
        agent.fragment_model_version = -1;
        agent.fragment_model_checksum.clear();
        agent.fragment_model_lineage_id.clear();
        agent.fragment_model_manifest_digest.clear();
        agent.fragment_first_action_frame_id = -1;
        agent.visited.clear();
        agent.visited.insert(
            session.start_gy * session.grid_cols + session.start_gx);
        agent.current_state_first_visit = false;
        agent.first_visit_bonus_total = 0.0f;
        const std::size_t start_index = static_cast<std::size_t>(
            session.start_gy * session.grid_cols + session.start_gx);
        agent.episode_start_geodesic_distance =
            start_index < session.geodesic_distance.size()
                ? session.geodesic_distance[start_index]
                : -1;
        agent.observation_grid_x = session.start_gx;
        agent.observation_grid_y = session.start_gy;
        agent.last_move_blocked = false;
        agent.blocked_move_count = 0;
        agent.observation_done = false;
        agent.last_observation_frame_id = -1;
        agent.episode_return = 0.0;
        agent.episode_transition_count = 0;
        agent.final_termination_reason =
            maze::MAZE_TERMINATION_REASON_UNSPECIFIED;
        agent.reward_component_sums.clear();
        if (config_.server.run_mode == aiserver_mode::kMapValidation) {
            agent.path_valid = agent.solver.PlanPath(
                session.start_gx, session.start_gy,
                session.end_gx, session.end_gy);
        }
    }
    RefreshFragmentSampleTarget(session);
}

bool MazeServiceImpl::ChooseModelAction(
    SessionManager::Session& session,
    SessionManager::AgentRuntime& agent,
    int gx,
    int gy,
    int64_t action_frame_id,
    int& action,
    float& log_prob,
    float& value) {
    return PrepareModelAction(
        session, agent, gx, gy, action_frame_id, nullptr,
        model_manifest_, action_rng_, action, log_prob, value);
}

bool MazeServiceImpl::PrepareModelAction(
    SessionManager::Session& session,
    SessionManager::AgentRuntime& agent,
    int gx,
    int gy,
    int64_t action_frame_id,
    const OnnxInferencer::PreparedModel* prepared_model,
    const ModelManifest& behavior_model,
    std::mt19937& action_rng,
    int& action,
    float& log_prob,
    float& value) {
    std::vector<float> obs;
    std::string observation_error;
    if (!MazeObservation::Build(
            session, agent, gx, gy, action_frame_id,
            config_.observation.ray_max_range,
            config_.model.expected_obs_dim,
            obs, observation_error)) {
        MarkDegraded("maze.observation.v3 construction failed: " +
                     observation_error);
        return false;
    }
    std::vector<float> logits;

    auto start = std::chrono::steady_clock::now();
    const bool inferred = prepared_model
        ? onnx_inferencer_.InferPrepared(
              *prepared_model, obs, static_cast<int>(obs.size()),
              logits, value)
        : onnx_inferencer_.Infer(
              obs, static_cast<int>(obs.size()), logits, value);
    double latency_ms = ElapsedMs(start);
    ++inference_count_;
    inference_latency_sum_ms_ += latency_ms;
    inference_latency_max_ms_ =
        std::max(inference_latency_max_ms_, latency_ms);

    if (!inferred) {
        MarkDegraded("ONNX inference failed");
        return false;
    }
    std::string model_output_error;
    if (!ValidateEpisodeModelOutput(
            logits, config_.model.expected_action_dim, value,
            model_output_error)) {
        MarkDegraded("ONNX model output is invalid: " + model_output_error);
        return false;
    }

    std::string action_error;
    if (!SelectEpisodeAction(
            logits, session.current_episode_mode,
            config_.policy.training_temperature, action_rng,
            action, log_prob, action_error)) {
        MarkDegraded(action_error);
        return false;
    }

    if (agent.has_pending_action) {
        MarkDegraded("Agent action would overwrite a pending transition");
        return false;
    }
    agent.pending_obs = std::move(obs);
    agent.pending_action = action;
    agent.pending_action_frame_id = action_frame_id;
    agent.pending_log_prob = log_prob;
    agent.pending_value = value;
    agent.pending_model_version = behavior_model.model_version;
    agent.pending_model_checksum = behavior_model.sha256;
    agent.pending_model_lineage_id = behavior_model.model_lineage_id;
    agent.pending_model_manifest_digest = behavior_model.manifest_digest;
    agent.has_pending_action = true;
    agent.prev_grid_x = gx;
    agent.prev_grid_y = gy;
    return true;
}

bool MazeServiceImpl::InferStateValue(
    const SessionManager::Session& session,
    const SessionManager::AgentRuntime& agent,
    int gx,
    int gy,
    int64_t episode_step,
    float& value) {
    return PrepareStateValue(
        session, agent, gx, gy, episode_step, nullptr, value);
}

bool MazeServiceImpl::PrepareStateValue(
    const SessionManager::Session& session,
    const SessionManager::AgentRuntime& agent,
    int gx,
    int gy,
    int64_t episode_step,
    const OnnxInferencer::PreparedModel* prepared_model,
    float& value) {
    std::vector<float> obs;
    std::string observation_error;
    if (!MazeObservation::Build(
            session, agent, gx, gy, episode_step,
            config_.observation.ray_max_range,
            config_.model.expected_obs_dim,
            obs, observation_error)) {
        MarkDegraded("maze.observation.v3 bootstrap construction failed: " +
                     observation_error);
        return false;
    }
    std::vector<float> probabilities;
    auto start = std::chrono::steady_clock::now();
    const bool inferred = prepared_model
        ? onnx_inferencer_.InferPrepared(
              *prepared_model, obs, static_cast<int>(obs.size()),
              probabilities, value)
        : onnx_inferencer_.Infer(
              obs, static_cast<int>(obs.size()), probabilities, value);
    const double latency_ms = ElapsedMs(start);
    ++inference_count_;
    inference_latency_sum_ms_ += latency_ms;
    inference_latency_max_ms_ =
        std::max(inference_latency_max_ms_, latency_ms);
    if (!inferred || !std::isfinite(value)) {
        MarkDegraded("ONNX bootstrap value inference failed");
        return false;
    }
    return true;
}

bool MazeServiceImpl::FinalizePendingTransition(
    SessionManager::Session& session,
    int agent_id,
    int gx,
    int gy,
    bool is_done,
    maze::MazeTerminationReason reason,
    bool collect_training_sample,
    int64_t& produced_unique_samples,
    std::unordered_map<int, int64_t>& produced_samples_by_model,
    std::string& error) {
    auto agent_it = session.agents.find(agent_id);
    if (agent_it == session.agents.end()) {
        error = "transition Agent identity is unknown";
        return false;
    }
    auto& agent = agent_it->second;
    if (!agent.has_pending_action) return true;
    const auto cache_it = session.agent_sample_caches.find(agent_id);
    const bool starts_new_fragment =
        collect_training_sample &&
        (cache_it == session.agent_sample_caches.end() ||
         cache_it->second.empty());
    if (collect_training_sample && !starts_new_fragment &&
        (agent.fragment_model_version != agent.pending_model_version ||
         agent.fragment_model_checksum != agent.pending_model_checksum ||
         agent.fragment_model_lineage_id != agent.pending_model_lineage_id ||
         agent.fragment_model_manifest_digest !=
             agent.pending_model_manifest_digest)) {
        error = "fragment contains mixed behavior models";
        return false;
    }

    RewardDetail reward;
    training::Sample sample;
    if (collect_training_sample) {
        reward = MazeReward::Calculate(
            session, agent_id, gx, gy, is_done, reason, config_.reward);
        if (!reward.valid) {
            error = reward.error;
            return false;
        }
        std::vector<float> next_observation;
        std::string observation_error;
        if (!MazeObservation::Build(
                session, agent, gx, gy,
                agent.pending_action_frame_id + 1,
                config_.observation.ray_max_range,
                config_.model.expected_obs_dim,
                next_observation, observation_error)) {
            error = "next observation construction failed: " +
                    observation_error;
            return false;
        }
        const std::vector<training::Sample> empty_fragment;
        const auto& fragment =
            cache_it == session.agent_sample_caches.end()
                ? empty_fragment
                : cache_it->second;
        if (!BuildTrainingSample(
                agent, next_observation, reward, is_done, reason,
                config_.model.expected_obs_dim,
                config_.model.expected_action_dim, fragment, sample,
                error)) {
            error = "training sample continuity is invalid: " + error;
            return false;
        }
    }

    // Commit only after all workload-specific validation has succeeded.
    // Reward V4 and training fragments belong exclusively to the training
    // workload. Standalone evaluation advances inference/lifecycle state only.
    if (starts_new_fragment) {
        agent.fragment_model_version = agent.pending_model_version;
        agent.fragment_model_checksum = agent.pending_model_checksum;
        agent.fragment_model_lineage_id = agent.pending_model_lineage_id;
        agent.fragment_model_manifest_digest =
            agent.pending_model_manifest_digest;
        agent.fragment_first_action_frame_id = agent.pending_action_frame_id;
    }
    ++agent.episode_transition_count;
    if (collect_training_sample) {
        agent.episode_return += reward.total;
        for (const auto& item : reward.items) {
            agent.reward_component_sums[item.first] += item.second;
            if (item.first == "first_visit_bonus") {
                agent.first_visit_bonus_total += item.second;
            }
        }
    }
    if (is_done) {
        agent.final_termination_reason = reason;
    }
    if (collect_training_sample) {
        auto& cache = session.agent_sample_caches[agent_id];
        cache.push_back(std::move(sample));
        ++produced_unique_samples;
        ++produced_samples_by_model[agent.pending_model_version];
    }
    agent.has_pending_action = false;
    agent.pending_action_frame_id = -1;
    agent.pending_model_version = -1;
    agent.pending_model_checksum.clear();
    agent.pending_model_lineage_id.clear();
    agent.pending_model_manifest_digest.clear();
    agent.pending_obs.clear();

    error.clear();
    return true;
}

bool MazeServiceImpl::ReconcileDiscardedTrainingSamples() {
    const auto sender = sample_sender_.GetSnapshot();
    if (sender.producer_stale_count < reconciled_producer_stale_samples_) {
        MarkDegraded("producer stale counter moved backwards");
        return false;
    }

    if (sender.pool_stale_count < reconciled_pool_stale_samples_) {
        MarkDegraded("sample pool stale counter moved backwards");
        return false;
    }

    int64_t producer_delta_total = 0;
    int64_t pool_delta_total = 0;
    std::unordered_map<int, int64_t> deltas;
    const auto collect_deltas = [this, &deltas](
        const std::unordered_map<int, int64_t>& current,
        const std::unordered_map<int, int64_t>& reconciled,
        const char* source,
        int64_t& delta_total) {
        for (const auto& item : current) {
            const auto previous = reconciled.find(item.first);
            const int64_t previous_value =
                previous == reconciled.end() ? 0 : previous->second;
            if (item.second < previous_value) {
                MarkDegraded(std::string(source) +
                             " behavior-model counter moved backwards");
                return false;
            }
            const int64_t delta = item.second - previous_value;
            if (delta == 0) continue;
            deltas[item.first] += delta;
            delta_total += delta;
        }
        return true;
    };
    if (!collect_deltas(sender.producer_stale_samples_by_model,
                        reconciled_producer_stale_samples_by_model_,
                        "producer stale", producer_delta_total) ||
        !collect_deltas(sender.pool_stale_samples_by_model,
                        reconciled_pool_stale_samples_by_model_,
                        "sample pool stale", pool_delta_total)) {
        return false;
    }
    if (reconciled_producer_stale_samples_ + producer_delta_total !=
            sender.producer_stale_count ||
        reconciled_pool_stale_samples_ + pool_delta_total !=
            sender.pool_stale_count) {
        MarkDegraded("discarded sample source accounting is inconsistent");
        return false;
    }

    const int64_t delta_total = producer_delta_total + pool_delta_total;
    for (const auto& item : deltas) {
        const int model_version = item.first;
        const auto produced = produced_samples_by_model_.find(model_version);
        if (produced == produced_samples_by_model_.end() ||
            produced->second < item.second) {
            MarkDegraded(
                "discarded sample behavior-model accounting is inconsistent");
            return false;
        }
    }
    if (produced_unique_samples_ < delta_total) {
        MarkDegraded("discarded sample accounting is inconsistent");
        return false;
    }
    std::string controller_error;
    if (delta_total > 0 &&
        !task_controller_.ReconcileDiscardedTrainingSamples(
            delta_total, controller_error)) {
        MarkDegraded("discarded sample task accounting is inconsistent: " +
                     controller_error);
        return false;
    }
    for (const auto& delta : deltas) {
        produced_samples_by_model_[delta.first] -= delta.second;
    }
    produced_unique_samples_ -= delta_total;
    reconciled_producer_stale_samples_ = sender.producer_stale_count;
    reconciled_producer_stale_samples_by_model_ =
        sender.producer_stale_samples_by_model;
    reconciled_pool_stale_samples_ = sender.pool_stale_count;
    reconciled_pool_stale_samples_by_model_ =
        sender.pool_stale_samples_by_model;
    return true;
}

void MazeServiceImpl::FillSampleBatchMetadata(
    training::SampleBatch& batch,
    const SessionManager::Session& session,
    int agent_id,
    maze::MazeTerminationReason,
    uint64_t sequence) {
    const auto agent_it = session.agents.find(agent_id);
    if (agent_it == session.agents.end()) return;
    const auto& agent = agent_it->second;
    batch.set_actor_session_id(session.session_id);
    batch.set_trajectory_id(session.current_episode_id);
    batch.set_actor_id(static_cast<uint32_t>(agent_id));
    batch.set_fragment_sequence(sequence);
    batch.set_fragment_id(static_cast<uint32_t>(
        sequence % std::numeric_limits<uint32_t>::max()));
    auto* policy = batch.mutable_behavior_policy();
    policy->set_model_lineage_id(agent.fragment_model_lineage_id);
    policy->set_model_version(
        static_cast<uint64_t>(agent.fragment_model_version));
    policy->set_distribution_schema_id(
        config_.policy.distribution_schema_id);
    FillDigest(config_.policy.policy_spec_digest,
               policy->mutable_policy_spec_digest());
    FillTrainingSemantics(config_, batch.mutable_training_semantics());
    aiserver_contract::FillSampleProducerIdentity(
        producer_instance_id_, 1, batch.mutable_producer());
    FillContract(config_, batch.mutable_contract());
    batch.set_created_at_unix_ms(NowMs());

    std::ostringstream batch_id;
    batch_id << producer_instance_id_ << "/"
             << session.session_id << "/"
             << session.current_episode_id << "/"
             << agent_id << "/" << sequence;
    batch.set_batch_id(batch_id.str());
}

bool MazeServiceImpl::FlushAgentSamples(
    SessionManager::Session& session,
    int agent_id,
    bool is_episode_end,
    maze::MazeTerminationReason reason,
    float bootstrap_value,
    bool bootstrap_valid) {
    auto& cache = session.agent_sample_caches[agent_id];
    if (cache.empty()) return true;
    auto agent_it = session.agents.find(agent_id);
    if (agent_it == session.agents.end()) return false;
    auto& agent = agent_it->second;

    auto pending_it = session.pending_sample_batches.find(agent_id);
    if (pending_it == session.pending_sample_batches.end()) {
        training::SampleBatch batch;
        batch.set_trajectory_end(is_episode_end);
        FillSampleBatchMetadata(
            batch, session, agent_id, reason,
            next_fragment_seq_.fetch_add(1));
        batch.set_bootstrap_value(bootstrap_value);
        batch.set_bootstrap_valid(bootstrap_valid);
        batch.set_first_action_step(static_cast<uint64_t>(
            agent.fragment_first_action_frame_id));
        batch.set_last_action_step(cache.back().action_step());
        for (const auto& sample : cache) {
            *batch.add_samples() = sample;
        }
        training::SampleBatch digest_source = batch;
        digest_source.clear_payload_digest();
        batch.mutable_payload_digest()->set_algorithm(
            common::DIGEST_ALGORITHM_SHA256);
        batch.mutable_payload_digest()->set_hex(
            Sha256Bytes(DeterministicBytes(digest_source)));
        if (batch.payload_digest().hex().empty()) {
            MarkDegraded("failed to calculate sample payload digest");
            return false;
        }
        pending_it =
            session.pending_sample_batches.emplace(agent_id, std::move(batch))
                .first;
        ++produced_unique_batches_;
    }

    auto start = std::chrono::steady_clock::now();
    bool enqueued = sample_sender_.Enqueue(pending_it->second);
    double latency_ms = ElapsedMs(start);
    ++enqueue_count_;
    enqueue_latency_sum_ms_ += latency_ms;
    enqueue_latency_max_ms_ = std::max(enqueue_latency_max_ms_, latency_ms);
    if (!enqueued) {
        auto sender = sample_sender_.GetSnapshot();
        MarkDegraded(sender.last_error.empty()
                         ? "failed to enqueue sample fragment"
                         : sender.last_error);
        return false;
    }

    cache.clear();
    session.pending_sample_batches.erase(pending_it);
    agent.fragment_model_version = -1;
    agent.fragment_model_checksum.clear();
    agent.fragment_model_lineage_id.clear();
    agent.fragment_model_manifest_digest.clear();
    agent.fragment_first_action_frame_id = -1;
    return true;
}

bool MazeServiceImpl::PrepareAgentSampleFlush(
    SessionManager::Session& session,
    int agent_id,
    bool is_episode_end,
    maze::MazeTerminationReason reason,
    float bootstrap_value,
    bool bootstrap_valid,
    uint64_t& next_fragment_sequence,
    int64_t& produced_unique_batches,
    std::vector<training::SampleBatch>& batches,
    std::string& error) {
    auto& cache = session.agent_sample_caches[agent_id];
    if (cache.empty()) return true;
    auto agent_it = session.agents.find(agent_id);
    if (agent_it == session.agents.end()) {
        error = "sample flush Agent identity is unknown";
        return false;
    }
    auto& agent = agent_it->second;

    auto pending_it = session.pending_sample_batches.find(agent_id);
    if (pending_it == session.pending_sample_batches.end()) {
        training::SampleBatch batch;
        batch.set_trajectory_end(is_episode_end);
        FillSampleBatchMetadata(
            batch, session, agent_id, reason, next_fragment_sequence);
        batch.set_bootstrap_value(bootstrap_value);
        batch.set_bootstrap_valid(bootstrap_valid);
        batch.set_first_action_step(static_cast<uint64_t>(
            agent.fragment_first_action_frame_id));
        batch.set_last_action_step(cache.back().action_step());
        for (const auto& sample : cache) {
            *batch.add_samples() = sample;
        }
        training::SampleBatch digest_source = batch;
        digest_source.clear_payload_digest();
        batch.mutable_payload_digest()->set_algorithm(
            common::DIGEST_ALGORITHM_SHA256);
        batch.mutable_payload_digest()->set_hex(
            Sha256Bytes(DeterministicBytes(digest_source)));
        if (batch.payload_digest().hex().empty()) {
            error = "failed to calculate sample payload digest";
            return false;
        }
        ++next_fragment_sequence;
        ++produced_unique_batches;
        pending_it = session.pending_sample_batches
                         .emplace(agent_id, std::move(batch))
                         .first;
    }

    batches.push_back(pending_it->second);
    cache.clear();
    session.pending_sample_batches.erase(pending_it);
    agent.fragment_model_version = -1;
    agent.fragment_model_checksum.clear();
    agent.fragment_model_lineage_id.clear();
    agent.fragment_model_manifest_digest.clear();
    agent.fragment_first_action_frame_id = -1;
    error.clear();
    return true;
}

void MazeServiceImpl::QuarantineAgentSamples(
    SessionManager::Session& session,
    int agent_id) {
    auto agent_it = session.agents.find(agent_id);
    if (agent_it == session.agents.end()) return;
    auto& agent = agent_it->second;
    auto& cache = session.agent_sample_caches[agent_id];
    const bool training_episode =
        session.current_episode_mode == maze::EPISODE_MODE_TRAINING;
    const int64_t discarded = training_episode
        ? static_cast<int64_t>(cache.size()) +
              (agent.has_pending_action ? 1 : 0)
        : 0;
    if (discarded > 0) {
        quarantined_sample_count_ += discarded;
        ++quarantined_fragment_count_;
    }
    if (!cache.empty()) {
        const int64_t cached_samples = static_cast<int64_t>(cache.size());
        const auto count = produced_samples_by_model_.find(
            agent.fragment_model_version);
        if (count == produced_samples_by_model_.end() ||
            count->second < cached_samples) {
            MarkDegraded(
                "quarantined behavior-model accounting is inconsistent");
        } else {
            count->second -= cached_samples;
        }
        produced_unique_samples_ -= cached_samples;
    }
    cache.clear();
    session.pending_sample_batches.erase(agent_id);
    agent.has_pending_action = false;
    agent.pending_action_frame_id = -1;
    agent.pending_model_version = -1;
    agent.pending_model_checksum.clear();
    agent.pending_model_lineage_id.clear();
    agent.pending_model_manifest_digest.clear();
    agent.pending_obs.clear();
    agent.fragment_model_version = -1;
    agent.fragment_model_checksum.clear();
    agent.fragment_model_lineage_id.clear();
    agent.fragment_model_manifest_digest.clear();
    agent.fragment_first_action_frame_id = -1;
}
