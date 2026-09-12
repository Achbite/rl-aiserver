#include "rl_sdk/metric_catalog.h"

#include "log/logger.h"
#include "task/sample/training_transition_builder.h"
#include "task/sample/rollout_transition_builder.h"
#include "task/policy/episode_action_policy.h"

#include <openssl/rand.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include "proto/metrics/catalog.pb.h"
#include "proto/metrics/registry.pb.h"
#include "proto/metrics/transport.pb.h"

namespace training_runtime_detail {

inline double ElapsedMs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

inline bool ShouldFetchModelCandidate(
    ModelStep latest_prepared_step,
    const std::optional<ModelStep>& staged_step,
    ModelStep distributor_latest_step) {
    return distributor_latest_step > latest_prepared_step &&
           (!staged_step.has_value() ||
            distributor_latest_step > *staged_step);
}

inline void FillServiceIdentity(const std::string& component,
                         const std::string& instance_id,
                         uint64_t lifecycle_epoch,
                         common::ServiceInstanceIdentity* target) {
    target->set_component(component);
    target->set_instance_id(instance_id);
    target->set_lifecycle_epoch(lifecycle_epoch);
}

inline common::ServiceInstanceIdentity MetricEventSource(
    const std::string& producer_instance_id,
    uint64_t lifecycle_epoch) {
    common::ServiceInstanceIdentity source;
    source.set_component("rl-aiserver");
    source.set_instance_id(producer_instance_id);
    source.set_lifecycle_epoch(lifecycle_epoch);
    return source;
}

}  // namespace training_runtime_detail


template <class Sessions>
TrainingRuntime<Sessions>::TrainingRuntime(
    const TrainingRuntimeConfig& config, std::string reward_metric_prefix)
    : config_(config),
      reward_metric_prefix_(std::move(reward_metric_prefix)),
      producer_instance_id_(
          CreateProducerInstanceId(config.sample_distributor.aiserver_id)),
      producer_lifecycle_epoch_(CreateProducerLifecycleEpoch()),
      sample_distributor_(config.sample_distributor),
      model_distributor_(config.model_distribution, config.model, producer_instance_id_,
                         producer_lifecycle_epoch_),
      next_lifecycle_epoch_(producer_lifecycle_epoch_ + 1),
      action_rng_(config.policy.sampling_seed),
      metric_events_(training_runtime_detail::MetricEventSource(producer_instance_id_,
                                       producer_lifecycle_epoch_)) {
}

template <class Sessions>
TrainingRuntime<Sessions>::~TrainingRuntime() { BeginShutdown(); }

template <class Sessions>
common::ServiceInstanceIdentity TrainingRuntime<Sessions>::MetricSourceIdentity() const {
    return training_runtime_detail::MetricEventSource(producer_instance_id_,
                             producer_lifecycle_epoch_);
}

template <class Sessions>
int64_t TrainingRuntime<Sessions>::NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

template <class Sessions>
bool TrainingRuntime<Sessions>::ValidateStagedModelProgress(
    const ModelManifest& active,
    const ModelManifest& candidate,
    std::string& error) const {
    if (!active.HasModelIdentity() || !candidate.HasModelIdentity() ||
        candidate.model_step() <= active.model_step() ||
        candidate.trained_samples() < active.trained_samples()) {
        error = "staged publication must advance without training-counter rollback";
        return false;
    }
    return true;
}

template <class Sessions>
std::string TrainingRuntime<Sessions>::CreateProducerInstanceId(
    const std::string& aiserver_id) {
    std::array<unsigned char, 16> nonce{};
    if (RAND_bytes(nonce.data(), static_cast<int>(nonce.size())) != 1) {
        throw std::runtime_error(
            "cannot generate AIServer producer instance identity");
    }
    std::ostringstream output;
    output << aiserver_id << "-" << NowMs() << "-" << getpid() << "-"
           << std::hex << std::setfill('0');
    for (const auto byte : nonce) {
        output << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return output.str();
}

template <class Sessions>
uint64_t TrainingRuntime<Sessions>::CreateProducerLifecycleEpoch() {
    std::array<unsigned char, sizeof(uint64_t)> nonce{};
    if (RAND_bytes(nonce.data(), static_cast<int>(nonce.size())) != 1) {
        throw std::runtime_error(
            "cannot generate AIServer producer lifecycle epoch");
    }
    uint64_t epoch = 0;
    for (const auto byte : nonce) {
        epoch = (epoch << 8) | static_cast<uint64_t>(byte);
    }
    return epoch == 0 ? 1 : epoch;
}

template <class Sessions>
bool TrainingRuntime<Sessions>::AcquireTrainingWorkspaceLease(std::string& error) {
    namespace fs = std::filesystem;
    if (config_.server.run_mode != aiserver_mode::kTraining ||
        training_workspace_lease_held_) {
        error.clear();
        return true;
    }

    const fs::path workspace(config_.model.local_train_dir);
    std::error_code filesystem_error;
    fs::create_directories(workspace, filesystem_error);
    if (filesystem_error) {
        error = "cannot create AIServer training workspace: " +
                filesystem_error.message();
        return false;
    }
    const auto workspace_status =
        fs::symlink_status(workspace, filesystem_error);
    if (filesystem_error || fs::is_symlink(workspace_status) ||
        !fs::is_directory(workspace_status)) {
        error = "AIServer training workspace must be a real directory";
        return false;
    }

    const fs::path lock_path = workspace / ".aiserver.lock";
    if (!fs::create_directory(lock_path, filesystem_error)) {
        error = filesystem_error
                    ? "cannot acquire AIServer training workspace lease: " +
                          filesystem_error.message()
                    : "AIServer training workspace is already in use";
        return false;
    }
    const fs::path pid_path = lock_path / "pid";
    {
        std::ofstream pid_file(pid_path, std::ios::trunc);
        pid_file << getpid() << '\n';
        pid_file.flush();
        if (!pid_file) {
            fs::remove_all(lock_path, filesystem_error);
            error = "cannot publish AIServer training workspace lease";
            return false;
        }
    }
    training_workspace_lock_path_ = lock_path.string();
    training_workspace_lease_held_ = true;
    error.clear();
    return true;
}

template <class Sessions>
bool TrainingRuntime<Sessions>::ReleaseTrainingWorkspaceLease(std::string& error) {
    namespace fs = std::filesystem;
    if (!training_workspace_lease_held_) {
        error.clear();
        return true;
    }

    const fs::path lock_path(training_workspace_lock_path_);
    std::error_code filesystem_error;
    fs::remove(lock_path / "pid", filesystem_error);
    if (filesystem_error) {
        error = "cannot remove AIServer training workspace lease PID: " +
                filesystem_error.message();
        return false;
    }
    if (!fs::remove(lock_path, filesystem_error) || filesystem_error) {
        error = filesystem_error
                    ? "cannot release AIServer training workspace lease: " +
                          filesystem_error.message()
                    : "AIServer training workspace lease is not empty";
        return false;
    }
    training_workspace_lock_path_.clear();
    training_workspace_lease_held_ = false;
    error.clear();
    return true;
}

template <class Sessions>
bool TrainingRuntime<Sessions>::LoadInitialModel() {
    model_state_.store(training::MODEL_STATE_WAITING);

    if (config_.server.run_mode == aiserver_mode::kEvaluation) {
        std::string error;
        const std::string model_path =
            config_.model.evaluation_model_path;
        std::error_code size_error;
        const auto model_size =
            std::filesystem::file_size(model_path, size_error);
        if (size_error || model_size == 0 ||
            model_size > static_cast<std::uintmax_t>(
                             std::numeric_limits<int64_t>::max())) {
            model_state_.store(training::MODEL_STATE_FAILED);
            last_error_ = "evaluation model file size is invalid";
            return false;
        }
        if (!onnx_inferencer_.LoadModel(
                model_path,
                config_.model.expected_obs_dim,
                config_.model.expected_action_dim, &error)) {
            model_state_.store(training::MODEL_STATE_FAILED);
            last_error_ = "evaluation model load failed: " + error;
            return false;
        }
        model_manifest_ = ModelManifest{};
        model_manifest_.wire.mutable_identity()->set_model_lineage_id(
            "evaluation-local");
        model_manifest_.wire.mutable_identity()->set_model_step(0);
        model_manifest_.wire.set_size_bytes(
            static_cast<int64_t>(model_size));
        model_manifest_.wire.set_published_at_unix_ms(NowMs());
        model_manifest_.model_path = model_path;
        model_state_.store(training::MODEL_STATE_READY);
        last_error_.clear();
        return true;
    }

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(config_.model.startup_timeout_ms);
    std::string error;
    while (std::chrono::steady_clock::now() < deadline) {
        ModelDistributorClient::AvailableRange range;
        if (model_distributor_.GetAvailableRange(
                config_.sample_distributor.aiserver_id, range, error)) {
            ModelManifest candidate;
            std::string load_error;
            OnnxInferencer::PreparedModel prepared;
            if (!FetchPrepareAndPublishModel(
                    range.latest_model_step, range.model_lineage_id, candidate, prepared,
                    load_error)) {
                model_state_.store(training::MODEL_STATE_FAILED);
                last_error_ = "initial model load failed: lineage=" +
                    range.model_lineage_id + " model_step=" +
                    std::to_string(range.latest_model_step) + ": " + load_error;
                std::string ack_error;
                const auto ack = model_distributor_.AckIdempotently(
                    candidate, config_.sample_distributor.aiserver_id,
                    training::MODEL_LOAD_STATUS_FAILED, last_error_, ack_error);
                if (ack != ModelDistributorClient::AckDisposition::Applied) {
                    last_error_ += "; FAILED acknowledgement was not confirmed: " +
                                   ack_error;
                }
                return false;
            }
            if (candidate.model_lineage_id() != range.model_lineage_id ||
                !prepared.valid()) {
                std::string ack_error;
                model_distributor_.Ack(
                    candidate, config_.sample_distributor.aiserver_id,
                    training::MODEL_LOAD_STATUS_FAILED,
                    "initial model range identity mismatch", ack_error);
                model_state_.store(training::MODEL_STATE_FAILED);
                last_error_ = "initial model range identity mismatch";
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
            std::string ack_error;
            auto ack = model_distributor_.AckIdempotently(
                candidate, config_.sample_distributor.aiserver_id,
                training::MODEL_LOAD_STATUS_LOADED, "loaded", ack_error,
                &ack_authority, deadline);
            if (ack == ModelDistributorClient::AckDisposition::Rejected ||
                ack == ModelDistributorClient::AckDisposition::NotApplied) {
                model_state_.store(training::MODEL_STATE_FAILED);
                last_error_ = "initial model ACK was rejected: " + ack_error;
                return false;
            }

            while (ack == ModelDistributorClient::AckDisposition::Uncertain &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                ack = model_distributor_.AckIdempotently(
                    candidate, config_.sample_distributor.aiserver_id,
                    training::MODEL_LOAD_STATUS_LOADED, "loaded", ack_error,
                    &ack_authority, deadline);
            }
            if (ack == ModelDistributorClient::AckDisposition::Rejected ||
                ack == ModelDistributorClient::AckDisposition::NotApplied) {
                model_state_.store(training::MODEL_STATE_FAILED);
                last_error_ = "initial model ACK was rejected: " + ack_error;
                return false;
            }

            if (ack != ModelDistributorClient::AckDisposition::Applied) {
                model_state_.store(training::MODEL_STATE_FAILED);
                last_error_ =
                    "initial model ACK remained outcome-unknown before "
                    "the startup deadline: " + ack_error;
                return false;
            }
            prepared.model_path = candidate.model_path;
            onnx_inferencer_.ActivatePreparedModel(std::move(prepared));
            model_manifest_ = candidate;
            std::string prune_error;
            if (!model_distributor_.PruneCache(
                    ProtectedCachedModelStepsLocked(), prune_error)) {
                LOG_ERROR("TrainingRuntime", "初始模型缓存淘汰延后: %s",
                          prune_error.c_str());
            }
            latest_prepared_used_by_agent_ = false;
            model_state_.store(training::MODEL_STATE_READY);
            last_error_.clear();
            LOG_INFO("TrainingRuntime",
                     "初始模型就绪: lineage=%s model_step=%llu",
                     model_manifest_.model_lineage_id().c_str(),
                     static_cast<unsigned long long>(
                         model_manifest_.model_step()));
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    model_state_.store(training::MODEL_STATE_FAILED);
    last_error_ = error.empty() ? "initial model startup timeout" : error;
    return false;
}

template <class Sessions>
bool TrainingRuntime<Sessions>::PrepareModelArtifact(
    const ModelManifest& manifest,
    OnnxInferencer::PreparedModel& prepared,
    std::string& error) {
    return onnx_inferencer_.PrepareModel(
        manifest.model_path, config_.model.expected_obs_dim,
        config_.model.expected_action_dim, prepared, &error);
}

template <class Sessions>
void TrainingRuntime<Sessions>::RecordModelFeedback(
    const training::ModelIdentity& candidate, const std::string& stage,
    const std::string& error) {
    std::lock_guard<std::mutex> lock(model_feedback_mutex_);
    *model_feedback_.mutable_candidate_model() = candidate;
    model_feedback_.set_stage(stage);
    model_feedback_.set_last_error(error);
}

template <class Sessions>
bool TrainingRuntime<Sessions>::FetchPrepareAndPublishModel(
    ModelStep model_step,
    const std::string& lineage_id,
    ModelManifest& manifest,
    OnnxInferencer::PreparedModel& prepared,
    std::string& error) {
    training::ModelIdentity target;
    target.set_model_lineage_id(lineage_id);
    target.set_model_step(model_step);
    manifest = ModelManifest{};
    *manifest.wire.mutable_identity() = target;
    ModelManifest downloaded;
    if (!model_distributor_.FetchStep(
            config_.sample_distributor.aiserver_id, model_step,
            downloaded, error)) {
        RecordModelFeedback(target, "download", error);
        error = "download: " + error;
        return false;
    }
    if (!PrepareModelArtifact(downloaded, prepared, error)) {
        std::string discard_error;
        model_distributor_.DiscardTemporary(downloaded, discard_error);
        if (!discard_error.empty()) error += "; " + discard_error;
        RecordModelFeedback(target, "prepare", error);
        error = "prepare: " + error;
        return false;
    }
    if (!model_distributor_.PublishPrepared(downloaded, error)) {
        std::string discard_error;
        model_distributor_.DiscardTemporary(downloaded, discard_error);
        if (!discard_error.empty()) error += "; " + discard_error;
        prepared = OnnxInferencer::PreparedModel{};
        RecordModelFeedback(target, "cache_publish", error);
        error = "cache_publish: " + error;
        return false;
    }
    RecordModelFeedback(target, "prepared", "");
    prepared.model_path = downloaded.model_path;
    manifest = std::move(downloaded);
    return true;
}

template <class Sessions>
std::set<ModelStep> TrainingRuntime<Sessions>::ProtectedCachedModelStepsLocked() {
    std::set<ModelStep> protected_steps;
    if (model_manifest_.HasModelIdentity()) {
        protected_steps.insert(model_manifest_.model_step());
    }
    if (staged_model_manifest_.HasModelIdentity()) {
        protected_steps.insert(staged_model_manifest_.model_step());
    }
    if (model_ack_pending_ &&
        pending_model_ack_manifest_.HasModelIdentity()) {
        protected_steps.insert(pending_model_ack_manifest_.model_step());
    }
    for (const auto& session_id : session_mgr_.GetSessionIds()) {
        const Session* session =
            session_mgr_.GetSession(session_id);
        if (!session) continue;
        for (const auto& item : session->agents) {
            const auto& agent = item.second;
            if (agent.segment_open &&
                agent.pinned_model.HasModelIdentity()) {
                protected_steps.insert(agent.pinned_model.model_step());
            }
        }
    }
    return protected_steps;
}

template <class Sessions>
void TrainingRuntime<Sessions>::StartModelWatcher() {
    if (config_.server.run_mode != aiserver_mode::kTraining ||
        model_watch_thread_.joinable()) {
        return;
    }
    model_watch_stop_.store(false);
    model_watch_thread_ =
        std::thread(&TrainingRuntime<Sessions>::ModelWatchLoop, this);
}

template <class Sessions>
void TrainingRuntime<Sessions>::StopModelWatcher() {
    model_watch_stop_.store(true);
    if (model_watch_thread_.joinable()) {
        model_watch_thread_.join();
    }
}

template <class Sessions>
void TrainingRuntime<Sessions>::RecordPendingModelAck(
    const ModelManifest& manifest,
    const common::ServiceInstanceIdentity& authority,
    std::chrono::steady_clock::time_point deadline,
    const std::string& error) {
    model_ack_pending_ = true;
    pending_model_ack_manifest_ = manifest;
    pending_model_ack_authority_ = authority;
    pending_model_ack_deadline_ = deadline;
    pending_model_ack_error_ = error;
    pending_model_ack_cause_ =
        "model ACK remains outcome-uncertain before latest-prepared publish: " +
        error;
    RecordModelFeedback(manifest.wire.identity(), "ack_pending",
                        pending_model_ack_cause_);
    LOG_WARN("TrainingRuntime", "%s", pending_model_ack_cause_.c_str());
}

template <class Sessions>
bool TrainingRuntime<Sessions>::RetryPendingModelAck() {
    ModelManifest pending;
    common::ServiceInstanceIdentity authority;
    std::chrono::steady_clock::time_point deadline;
    std::string ack_error;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!model_ack_pending_) return true;
        if (model_state_.load() == training::MODEL_STATE_FAILED) return false;
        pending = pending_model_ack_manifest_;
        authority = pending_model_ack_authority_;
        deadline = pending_model_ack_deadline_;
        ack_error = pending_model_ack_error_;
    }

    const auto ack = model_distributor_.AckIdempotently(
        pending, config_.sample_distributor.aiserver_id,
        training::MODEL_LOAD_STATUS_LOADED, "loaded", ack_error,
        &authority, deadline);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!model_ack_pending_ ||
        pending_model_ack_manifest_.model_step() != pending.model_step() ||
        pending_model_ack_manifest_.model_lineage_id() !=
            pending.model_lineage_id()) {
        return !model_ack_pending_;
    }
    if (ack == ModelDistributorClient::AckDisposition::Applied) {
        if (!staged_model_manifest_.HasModelIdentity() ||
            staged_model_manifest_.model_step() != pending.model_step() ||
            staged_model_manifest_.model_lineage_id() !=
                pending.model_lineage_id() ||
            !staged_prepared_model_.valid()) {
            model_state_.store(training::MODEL_STATE_FAILED);
            state_.store(training::AISERVER_STATE_DEGRADED);
            last_error_ =
                "acknowledged staged model is no longer available locally";
            return false;
        }
        if (model_manifest_.HasModelIdentity() &&
            !latest_prepared_used_by_agent_) {
            ++superseded_without_agent_activation_count_;
        }
        staged_prepared_model_.model_path =
            staged_model_manifest_.model_path;
        onnx_inferencer_.ActivatePreparedModel(
            std::move(staged_prepared_model_));
        model_manifest_ = staged_model_manifest_;
        staged_model_manifest_ = ModelManifest{};
        staged_prepared_model_ = OnnxInferencer::PreparedModel{};
        ++model_switch_count_;
        latest_prepared_used_by_agent_ = false;
        model_ack_pending_ = false;
        pending_model_ack_manifest_ = ModelManifest{};
        pending_model_ack_authority_.Clear();
        pending_model_ack_error_.clear();
        pending_model_ack_cause_.clear();
        RecordModelFeedback(model_manifest_.wire.identity(), "loaded", "");
        model_state_.store(training::MODEL_STATE_READY);
        LOG_INFO("TrainingRuntime",
                 "模型已成为 latest-prepared: lineage=%s model_step=%llu",
                 model_manifest_.model_lineage_id().c_str(),
                 static_cast<unsigned long long>(model_manifest_.model_step()));
        return true;
    }
    if (ack == ModelDistributorClient::AckDisposition::Rejected ||
        ack == ModelDistributorClient::AckDisposition::NotApplied) {
        model_ack_pending_ = false;
        pending_model_ack_error_ = ack_error;
        model_state_.store(training::MODEL_STATE_FAILED);
        state_.store(training::AISERVER_STATE_DEGRADED);
        last_error_ = "pending model ACK was explicitly rejected: " +
                      ack_error;
        RecordModelFeedback(pending.wire.identity(), "ack_rejected", last_error_);
        return false;
    }
    if (!ack_error.empty()) pending_model_ack_error_ = ack_error;
    if (std::chrono::steady_clock::now() >= deadline) {
        model_state_.store(training::MODEL_STATE_FAILED);
        pending_model_ack_cause_ =
            "model ACK recovery deadline elapsed with outcome unknown: lineage=" +
            pending.model_lineage_id() + " model_step=" +
            std::to_string(pending.model_step()) + " authority=" +
            authority.instance_id() + ": " + pending_model_ack_error_;
        state_.store(training::AISERVER_STATE_DEGRADED);
        last_error_ = pending_model_ack_cause_;
        LOG_ERROR("TrainingRuntime", "%s", last_error_.c_str());
        RecordModelFeedback(pending.wire.identity(), "ack_unconfirmed", last_error_);
        return false;
    }
    pending_model_ack_cause_ =
        "pending model ACK remains outcome-uncertain: " + pending_model_ack_error_;
    RecordModelFeedback(pending.wire.identity(), "ack_pending",
                        pending_model_ack_cause_);
    return false;
}

template <class Sessions>
bool TrainingRuntime<Sessions>::TryPromoteStagedModelForWatcher() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_started_ || model_ack_pending_ ||
        !staged_model_manifest_.HasModelIdentity() ||
        !staged_prepared_model_.valid()) {
        return false;
    }
    return ActivateStagedModel();
}

template <class Sessions>
bool TrainingRuntime<Sessions>::RecoverExpiredClientSessions() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_started_ ||
        config_.server.run_mode != aiserver_mode::kTraining) {
        return true;
    }

    const int64_t now = NowMs();
    for (const auto& session_id : session_mgr_.GetSessionIds()) {
        auto* session = session_mgr_.GetSession(session_id);
        if (!session ||
            (session->phase != rl::session::v1::SESSION_PHASE_EPISODE_RUNNING &&
             session->phase != rl::session::v1::SESSION_PHASE_EPISODE_TERMINAL) ||
            session->last_valid_client_activity_unix_ms <= 0 ||
            now < session->last_valid_client_activity_unix_ms ||
            now - session->last_valid_client_activity_unix_ms <
                config_.sample_distributor.recovery_timeout_ms) {
            continue;
        }

        Session candidate = *session;
        int64_t candidate_produced_transitions =
            produced_unique_transitions_;
        int64_t candidate_produced_envelopes =
            produced_unique_envelopes_;
        auto candidate_produced_by_model =
            produced_transitions_by_model_;
        int64_t candidate_quarantined_transitions =
            quarantined_transition_count_;
        int64_t candidate_pending_actions_excluded =
            pending_action_excluded_count_;
        int64_t candidate_closed_segments =
            closed_segment_count_;
        auto candidate_close_counts = segment_close_counts_;
        std::vector<training::ProcessedTransitionEnvelope> envelopes;
        std::string error;
        bool preparation_ok = true;

        for (auto& item : candidate.agents) {
            auto& agent = item.second;
            if (!agent.segment_open && !agent.has_pending_action &&
                agent.segment_transitions.empty()) {
                continue;
            }
            if (agent.segment_open &&
                !agent.segment_transitions.empty()) {
                const bool had_pending_action =
                    agent.has_pending_action;
                float bootstrap_value = 0.0f;
                bool bootstrap_valid = true;
                if (had_pending_action) {
                    bootstrap_value = agent.pending_value;
                    bootstrap_valid =
                        std::isfinite(bootstrap_value);
                } else {
                    bootstrap_valid = InferPinnedValue(
                        agent,
                        agent.segment_transitions.back()
                            .next_observation,
                        bootstrap_value);
                }
                if (!bootstrap_valid ||
                    !PrepareAgentSegmentClose(
                        candidate, item.first,
                        training::
                            SEGMENT_CLOSE_REASON_CLIENT_RECOVERY_TIMEOUT,
                        bootstrap_value, true,
                        candidate_produced_transitions,
                        candidate_produced_envelopes,
                        candidate_produced_by_model, envelopes,
                        candidate_close_counts, error)) {
                    ++rollout_estimator_failure_count_;
                    if (error.empty()) {
                        error =
                            "client recovery timeout has no finite pinned "
                            "bootstrap";
                    }
                    preparation_ok = false;
                    break;
                }
                ++candidate_closed_segments;
                if (had_pending_action) {
                    ++candidate_pending_actions_excluded;
                    agent.has_pending_action = false;
                    agent.pending_action_frame_id = -1;
                    agent.pending_obs.clear();
                    agent.pending_action_mask.clear();
                }
            } else {
                DiscardAgentSegment(
                    candidate, item.first,
                    training::
                        SEGMENT_CLOSE_REASON_CLIENT_RECOVERY_TIMEOUT,
                    true, candidate_quarantined_transitions,
                    candidate_pending_actions_excluded,
                    candidate_closed_segments,
                    candidate_close_counts);
            }
        }

        if (!preparation_ok) {
            for (auto& item : session->agents) {
                DiscardAgentSegment(
                    *session, item.first,
                    training::
                        SEGMENT_CLOSE_REASON_CLIENT_RECOVERY_TIMEOUT,
                    true, quarantined_transition_count_,
                    pending_action_excluded_count_,
                    closed_segment_count_, segment_close_counts_);
            }
            session->phase = rl::session::v1::SESSION_PHASE_ABORTED;
            session->behavior_policy_scope =
                BehaviorPolicyScope::Unspecified;
            MarkDegraded(
                "client recovery timeout segment close failed: " + error);
            return false;
        }

        uint64_t reservation_id = 0;
        const auto reservation =
            sample_distributor_.ReserveEnqueueEnvelopeSet(
                envelopes, reservation_id, error);
        if (reservation ==
            SampleDistributor::ReservationResult::kRetryableUnavailable) {
            LOG_WARN(
                "TrainingRuntime",
                "Client recovery timeout output waits for SampleDistributor: session=%s cause=%s",
                session_id.c_str(), error.c_str());
            continue;
        }
        if (reservation ==
            SampleDistributor::ReservationResult::kTerminalFault) {
            MarkDegraded(
                "client recovery timeout reservation failed: " + error);
            return false;
        }
        const auto seal =
            sample_distributor_.SealEnqueueEnvelopeSet(
                reservation_id, error);
        if (seal ==
            SampleDistributor::SealResult::kRetryableUnavailable) {
            LOG_WARN(
                "TrainingRuntime",
                "Client recovery timeout output seal waits: session=%s cause=%s",
                session_id.c_str(), error.c_str());
            continue;
        }
        if (seal ==
            SampleDistributor::SealResult::kTerminalFault) {
            MarkDegraded(
                "client recovery timeout seal failed: " + error);
            return false;
        }
        if (sample_distributor_.CommitEnqueueEnvelopeSet(
                reservation_id, error) !=
            SampleDistributor::CommitResult::kCommitted) {
            MarkDegraded(
                "client recovery timeout envelope commit failed: " +
                error);
            return false;
        }

        candidate.phase = rl::session::v1::SESSION_PHASE_ABORTED;
        candidate.behavior_policy_scope =
            BehaviorPolicyScope::Unspecified;
        candidate.evaluation_pinned_model_lineage_id.clear();
        candidate.evaluation_pinned_model_step = 0;
        *session = std::move(candidate);
        produced_unique_transitions_ =
            candidate_produced_transitions;
        produced_unique_envelopes_ =
            candidate_produced_envelopes;
        produced_transitions_by_model_ =
            std::move(candidate_produced_by_model);
        quarantined_transition_count_ =
            candidate_quarantined_transitions;
        pending_action_excluded_count_ =
            candidate_pending_actions_excluded;
        closed_segment_count_ = candidate_closed_segments;
        segment_close_counts_ = std::move(candidate_close_counts);
        LOG_WARN(
            "TrainingRuntime",
            "Client recovery timeout closed active Episode: session=%s episode=%s idle_ms=%lld",
            session_id.c_str(), session->current_episode_id.c_str(),
            static_cast<long long>(
                now - session->last_valid_client_activity_unix_ms));
    }
    return true;
}

template <class Sessions>
void TrainingRuntime<Sessions>::ModelWatchLoop() {
    const auto poll_interval = std::chrono::milliseconds(
        config_.model_distribution.poll_interval_ms);
    std::string last_reported_watch_error;
    while (!model_watch_stop_.load()) {
        RecoverExpiredClientSessions();
        bool has_pending_ack = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            has_pending_ack = model_ack_pending_;
        }
        if (model_state_.load() != training::MODEL_STATE_FAILED && has_pending_ack) {
            RetryPendingModelAck();
        } else if (model_state_.load() != training::MODEL_STATE_FAILED) {
            ModelStep active_step = 0;
            std::optional<ModelStep> staged_step;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                active_step = model_manifest_.model_step();
                if (staged_model_manifest_.HasModelIdentity()) {
                    staged_step = staged_model_manifest_.model_step();
                }
            }

            ModelDistributorClient::AvailableRange range;
            std::string error;
            std::string cycle_error;
            bool range_ready = model_distributor_.GetAvailableRange(
                config_.sample_distributor.aiserver_id, range, error);
            if (!range_ready) {
                RecordModelFeedback({}, "discover", error);
            } else {
                std::lock_guard<std::mutex> feedback_lock(model_feedback_mutex_);
                if (model_feedback_.stage() == "discover") {
                    model_feedback_.set_stage("available");
                    model_feedback_.clear_last_error();
                    model_feedback_.mutable_candidate_model()->set_model_lineage_id(range.model_lineage_id);
                    model_feedback_.mutable_candidate_model()->set_model_step(range.latest_model_step);
                }
            }
            bool latest_ready = range_ready;
            if (range_ready &&
                training_runtime_detail::ShouldFetchModelCandidate(
                    active_step, staged_step,
                    range.latest_model_step)) {
                ModelManifest candidate;
                OnnxInferencer::PreparedModel prepared;
                latest_ready = FetchPrepareAndPublishModel(
                    range.latest_model_step, range.model_lineage_id, candidate, prepared,
                    error);
                if (latest_ready &&
                    candidate.model_lineage_id() != range.model_lineage_id) {
                    error = "staged model range identity changed during fetch";
                    RecordModelFeedback(candidate.wire.identity(), "range_identity", error);
                    latest_ready = false;
                }
                if (latest_ready) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    const std::optional<ModelStep> current_staged =
                        staged_model_manifest_.HasModelIdentity()
                            ? std::optional<ModelStep>(
                                  staged_model_manifest_.model_step())
                            : std::nullopt;
                    if (training_runtime_detail::ShouldFetchModelCandidate(
                            model_manifest_.model_step(),
                            current_staged,
                            candidate.model_step())) {
                        staged_model_manifest_ = std::move(candidate);
                        staged_prepared_model_ = std::move(prepared);
                        LOG_INFO(
                            "TrainingRuntime",
                            "模型已暂存: lineage=%s model_step=%llu",
                            staged_model_manifest_.model_lineage_id().c_str(),
                            static_cast<unsigned long long>(
                                staged_model_manifest_.model_step()));
                    }
                }
            }
            TryPromoteStagedModelForWatcher();
            if (model_state_.load() != training::MODEL_STATE_FAILED) {
                bool has_pending_ack_after_activation = false;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    has_pending_ack_after_activation = model_ack_pending_;
                }
                if ((!range_ready || !latest_ready ||
                     has_pending_ack_after_activation) &&
                    !error.empty()) {
                    cycle_error = "model refresh delayed: " + error;
                }
                std::set<ModelStep> protected_steps;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    protected_steps = ProtectedCachedModelStepsLocked();
                }
                std::string prune_error;
                if (!model_distributor_.PruneCache(
                        protected_steps, prune_error)) {
                    if (!cycle_error.empty()) cycle_error += "; ";
                    cycle_error += "model cache pruning delayed: " + prune_error;
                }
                if (!cycle_error.empty() &&
                    cycle_error != last_reported_watch_error) {
                    LOG_ERROR("TrainingRuntime", "%s", cycle_error.c_str());
                    last_reported_watch_error = cycle_error;
                } else if (cycle_error.empty() &&
                           !last_reported_watch_error.empty()) {
                    LOG_INFO("TrainingRuntime", "模型刷新与缓存维护已恢复");
                    last_reported_watch_error.clear();
                }
            }
        }

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

template <class Sessions>
bool TrainingRuntime<Sessions>::Start() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (started_) return state_.load() == training::AISERVER_STATE_READY;

        state_.store(training::AISERVER_STATE_STARTING);
        if (config_.server.run_mode == aiserver_mode::kTraining) {
            std::string workspace_error;
            if (!AcquireTrainingWorkspaceLease(workspace_error)) {
                last_error_ = workspace_error;
                state_.store(training::AISERVER_STATE_DEGRADED);
                LOG_ERROR("TrainingRuntime", "训练 workspace 独占失败: %s",
                          last_error_.c_str());
                return false;
            }
            if (!sample_distributor_.Start()) {
                auto sender = sample_distributor_.GetSnapshot();
                last_error_ = sender.last_error;
                std::string release_error;
                if (!ReleaseTrainingWorkspaceLease(release_error)) {
                    last_error_ += "; " + release_error;
                }
                state_.store(training::AISERVER_STATE_DEGRADED);
                LOG_ERROR("TrainingRuntime", "样本链路启动失败: %s",
                          last_error_.c_str());
                return false;
            }
            if (!LoadInitialModel()) {
                sample_distributor_.StopAndDrain();
                std::string release_error;
                if (!ReleaseTrainingWorkspaceLease(release_error)) {
                    last_error_ += "; " + release_error;
                }
                state_.store(training::AISERVER_STATE_DEGRADED);
                LOG_ERROR("TrainingRuntime", "训练模型加载失败: %s",
                          last_error_.c_str());
                return false;
            }
        } else if (config_.server.run_mode == aiserver_mode::kEvaluation) {
            if (!LoadInitialModel()) {
                state_.store(training::AISERVER_STATE_DEGRADED);
                LOG_ERROR("TrainingRuntime", "启动模型加载失败: %s",
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
    if (config_.server.run_mode == aiserver_mode::kTraining) {
        metric_flush_.Start(std::chrono::milliseconds(config_.reward_metric_interval_ms), [this] {
            std::lock_guard<std::mutex> lock(mutex_);
            return FlushRewardMetricsLocked();
        });
    }
    return true;
}

template <class Sessions>
bool TrainingRuntime<Sessions>::IsReady() const {
    return IsCoreInferenceReady() &&
           (config_.server.run_mode != aiserver_mode::kTraining ||
            sample_distributor_.IsTrainingDeliveryReady());
}

template <class Sessions>
bool TrainingRuntime<Sessions>::IsCoreInferenceReady() const {
    return state_.load() == training::AISERVER_STATE_READY &&
           model_state_.load() == training::MODEL_STATE_READY;
}

template <class Sessions>
bool TrainingRuntime<Sessions>::BeginShutdown() {
    metric_flush_.Stop();
    StopModelWatcher();

    bool preexisting_service_fault = false;
    std::string preexisting_service_error;
    bool close_preparation_ok = true;
    std::string close_preparation_error;
    std::vector<training::ProcessedTransitionEnvelope> envelopes;
    std::unordered_map<std::string, Session>
        candidate_sessions;
    int64_t candidate_produced_transitions = 0;
    int64_t candidate_produced_envelopes = 0;
    std::unordered_map<ModelStep, int64_t> candidate_produced_by_model;
    int64_t candidate_quarantined_transitions = 0;
    int64_t candidate_pending_actions_excluded = 0;
    int64_t candidate_closed_segments = 0;
    std::unordered_map<training::SegmentCloseReason, int64_t>
        candidate_close_counts;

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

        candidate_produced_transitions = produced_unique_transitions_;
        candidate_produced_envelopes = produced_unique_envelopes_;
        candidate_produced_by_model = produced_transitions_by_model_;
        candidate_quarantined_transitions =
            quarantined_transition_count_;
        candidate_pending_actions_excluded =
            pending_action_excluded_count_;
        candidate_closed_segments = closed_segment_count_;
        candidate_close_counts = segment_close_counts_;

        for (const auto& session_id : session_mgr_.GetSessionIds()) {
            auto* current = session_mgr_.GetSession(session_id);
            if (!current) continue;
            Session candidate = *current;
            const bool training_episode =
                config_.server.run_mode == aiserver_mode::kTraining &&
                candidate.current_episode_mode ==
                    PolicyMode::Training;
            for (auto& item : candidate.agents) {
                auto& agent = item.second;
                if (!training_episode) {
                    agent.has_pending_action = false;
                    agent.pending_action_frame_id = -1;
                    agent.pending_obs.clear();
                    agent.pending_action_mask.clear();
                    continue;
                }
                if (!agent.segment_open && !agent.has_pending_action &&
                    agent.segment_transitions.empty()) {
                    continue;
                }
                if (agent.segment_open &&
                    !agent.segment_transitions.empty()) {
                    const bool had_pending_action =
                        agent.has_pending_action;
                    float bootstrap_value = 0.0f;
                    bool bootstrap_valid = true;
                    if (had_pending_action) {
                        bootstrap_value = agent.pending_value;
                        bootstrap_valid =
                            std::isfinite(bootstrap_value);
                    } else {
                        bootstrap_valid = InferPinnedValue(
                            agent,
                            agent.segment_transitions.back()
                                .next_observation,
                            bootstrap_value);
                    }
                    std::string error;
                    if (!bootstrap_valid ||
                        !PrepareAgentSegmentClose(
                            candidate, item.first,
                            training::
                                SEGMENT_CLOSE_REASON_AISERVER_CONTROLLED_SHUTDOWN,
                            bootstrap_value, true,
                            candidate_produced_transitions,
                            candidate_produced_envelopes,
                            candidate_produced_by_model, envelopes,
                            candidate_close_counts, error)) {
                        ++rollout_estimator_failure_count_;
                        if (close_preparation_error.empty()) {
                            close_preparation_error =
                                bootstrap_valid
                                    ? "shutdown Agent segment close failed: " +
                                          error
                                    : "shutdown Agent segment has no finite "
                                      "pinned bootstrap";
                        }
                        close_preparation_ok = false;
                        break;
                    }
                    ++candidate_closed_segments;
                    if (had_pending_action) {
                        ++candidate_pending_actions_excluded;
                        agent.has_pending_action = false;
                        agent.pending_action_frame_id = -1;
                        agent.pending_obs.clear();
                        agent.pending_action_mask.clear();
                    }
                } else {
                    DiscardAgentSegment(
                        candidate, item.first,
                        training::
                            SEGMENT_CLOSE_REASON_AISERVER_CONTROLLED_SHUTDOWN,
                        true, candidate_quarantined_transitions,
                        candidate_pending_actions_excluded,
                        candidate_closed_segments,
                        candidate_close_counts);
                }
            }
            if (!close_preparation_ok) break;
            if (candidate.phase == rl::session::v1::SESSION_PHASE_EPISODE_RUNNING ||
                candidate.phase == rl::session::v1::SESSION_PHASE_EPISODE_TERMINAL) {
                candidate.phase = rl::session::v1::SESSION_PHASE_ABORTED;
                candidate.behavior_policy_scope =
                    BehaviorPolicyScope::Unspecified;
                candidate.evaluation_pinned_model_lineage_id.clear();
                candidate.evaluation_pinned_model_step = 0;
            }
            candidate_sessions.emplace(
                session_id, std::move(candidate));
        }
    }

    bool close_transaction_committed =
        config_.server.run_mode != aiserver_mode::kTraining ||
        envelopes.empty();
    if (config_.server.run_mode == aiserver_mode::kTraining &&
        close_preparation_ok && !envelopes.empty()) {
        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(
                config_.sample_distributor.recovery_timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            uint64_t reservation_id = 0;
            std::string error;
            const auto reservation =
                sample_distributor_.ReserveEnqueueEnvelopeSet(
                    envelopes, reservation_id, error);
            if (reservation ==
                SampleDistributor::ReservationResult::kTerminalFault) {
                close_preparation_error =
                    "shutdown envelope reservation failed: " + error;
                break;
            }
            if (reservation ==
                SampleDistributor::ReservationResult::
                    kRetryableUnavailable) {
                close_preparation_error =
                    error.empty()
                        ? "shutdown envelope reservation is temporarily "
                          "unavailable"
                        : error;
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(50));
                continue;
            }

            const auto seal =
                sample_distributor_.SealEnqueueEnvelopeSet(
                    reservation_id, error);
            if (seal ==
                SampleDistributor::SealResult::kTerminalFault) {
                close_preparation_error =
                    "shutdown envelope seal failed: " + error;
                break;
            }
            if (seal ==
                SampleDistributor::SealResult::
                    kRetryableUnavailable) {
                close_preparation_error =
                    error.empty()
                        ? "shutdown envelope seal is temporarily unavailable"
                        : error;
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(50));
                continue;
            }

            if (sample_distributor_.CommitEnqueueEnvelopeSet(
                    reservation_id, error) !=
                SampleDistributor::CommitResult::kCommitted) {
                close_preparation_error =
                    "shutdown sealed envelope commit failed: " + error;
                break;
            }
            close_transaction_committed = true;
            close_preparation_error.clear();
            break;
        }
        if (!close_transaction_committed &&
            close_preparation_error.empty()) {
            close_preparation_error =
                "shutdown envelope recovery deadline expired";
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (close_preparation_ok && close_transaction_committed) {
            for (auto& item : candidate_sessions) {
                auto* current = session_mgr_.GetSession(item.first);
                if (current) *current = std::move(item.second);
            }
            produced_unique_transitions_ =
                candidate_produced_transitions;
            produced_unique_envelopes_ =
                candidate_produced_envelopes;
            produced_transitions_by_model_ =
                std::move(candidate_produced_by_model);
            quarantined_transition_count_ =
                candidate_quarantined_transitions;
            pending_action_excluded_count_ =
                candidate_pending_actions_excluded;
            closed_segment_count_ = candidate_closed_segments;
            segment_close_counts_ =
                std::move(candidate_close_counts);
        } else {
            quarantined_envelope_count_ +=
                static_cast<int64_t>(envelopes.size());
            for (const auto& session_id :
                 session_mgr_.GetSessionIds()) {
                auto* session = session_mgr_.GetSession(session_id);
                if (!session) continue;
                for (auto& item : session->agents) {
                    DiscardAgentSegment(
                        *session, item.first,
                        training::
                            SEGMENT_CLOSE_REASON_AISERVER_CONTROLLED_SHUTDOWN,
                        true, quarantined_transition_count_,
                        pending_action_excluded_count_,
                        closed_segment_count_,
                        segment_close_counts_);
                }
            }
            if (!close_preparation_error.empty()) {
                last_error_ = close_preparation_error;
            }
        }
        if (!FlushRewardMetricsLocked()) {
            close_preparation_ok = false;
            close_preparation_error = "reward tail was not accepted by the local journal";
            last_error_ = close_preparation_error;
        }
        metric_events_.Finalize();
    }

    bool sender_drained = true;
    if (config_.server.run_mode == aiserver_mode::kTraining) {
        sender_drained = sample_distributor_.StopAndDrain();
    }
    const bool metric_events_settled =
        config_.server.run_mode != aiserver_mode::kTraining ||
        metric_events_.WaitForFinalAcknowledgement(
            std::chrono::seconds(10));
    if (!metric_events_settled) {
        LOG_ERROR(
            "TrainingRuntime",
            "AIServer metric source final batch was not acknowledged before "
            "the shutdown deadline");
    }
    std::string workspace_release_error;
    const bool workspace_released =
        ReleaseTrainingWorkspaceLease(workspace_release_error);

    std::lock_guard<std::mutex> lock(mutex_);
    const auto sender = sample_distributor_.GetSnapshot();
    const bool shutdown_accounting_fault =
        !close_preparation_ok || !close_transaction_committed ||
        state_.load() == training::AISERVER_STATE_DEGRADED;
    const std::string shutdown_accounting_error = last_error_;
    const bool model_state_settled =
        !model_ack_pending_ &&
        model_state_.load() != training::MODEL_STATE_FAILED &&
        (!started_ ||
         model_state_.load() == training::MODEL_STATE_READY);
    shutdown_succeeded_ =
        !preexisting_service_fault && !shutdown_accounting_fault &&
        model_state_settled && sender_drained &&
        metric_events_settled && workspace_released &&
        !sender.degraded && !sender.terminal_fault &&
        sender.queue_envelopes == 0 &&
        sender.queue_transitions == 0 &&
        sender.unresolved_push_outcome_unknown_transitions == 0 &&
        sender.unresolved_push_outcome_unknown_envelopes == 0 &&
        sender.final_drop_unique_transitions == 0 &&
        sender.final_drop_unique_envelopes == 0 &&
        CountCachedTransitions() == 0 &&
        CountCachedSegments() == 0;
    shutdown_completed_ = true;
    if (shutdown_succeeded_) {
        state_.store(training::AISERVER_STATE_STOPPED);
        return true;
    }

    std::ostringstream error;
    error << "AIServer shutdown did not settle: drained="
          << (sender_drained ? 1 : 0)
          << " outbound_transitions=" << sender.queue_transitions
          << " outbound_envelopes=" << sender.queue_envelopes
          << " cached_transitions=" << CountCachedTransitions()
          << " cached_segments=" << CountCachedSegments()
          << " unresolved_push_transitions="
          << sender.unresolved_push_outcome_unknown_transitions
          << " unresolved_push_envelopes="
          << sender.unresolved_push_outcome_unknown_envelopes
          << " final_drop_transitions="
          << sender.final_drop_unique_transitions
          << " final_drop_envelopes="
          << sender.final_drop_unique_envelopes
          << " model_ack_pending=" << (model_ack_pending_ ? 1 : 0)
          << " model_state=" << static_cast<int>(model_state_.load())
          << " metric_final_ack="
          << (metric_events_settled ? 1 : 0)
          << " workspace_released="
          << (workspace_released ? 1 : 0);
    const std::string& preserved_error =
        !shutdown_accounting_error.empty()
            ? shutdown_accounting_error
            : preexisting_service_error;
    if (!preserved_error.empty()) {
        error << " cause=" << preserved_error;
    }
    if (!workspace_release_error.empty()) {
        error << " workspace_error=" << workspace_release_error;
    }
    last_error_ = error.str();
    state_.store(training::AISERVER_STATE_DEGRADED);
    LOG_ERROR("TrainingRuntime", "%s", last_error_.c_str());
    return false;
}

template <class Sessions>
bool TrainingRuntime<Sessions>::EnsureAgentSegmentPin(
    Session& session,
    AgentRuntime& agent,
    int agent_id,
    uint64_t& next_segment_sequence,
    int64_t& per_agent_activation_count,
    bool& latest_model_used,
    std::string& error) {
    if (agent.segment_open) {
        if (!agent.pinned_model.HasModelIdentity() ||
            !agent.pinned_prepared_model.valid() ||
            agent.segment_id.empty()) {
            error = "open Agent segment has incomplete pinned model state";
            return false;
        }
        return true;
    }
    if (!model_manifest_.HasModelIdentity()) {
        error = "latest prepared training model has no identity";
        return false;
    }
    auto prepared = onnx_inferencer_.SnapshotPreparedModel();
    if (!prepared.valid() ||
        prepared.model_path != model_manifest_.model_path) {
        error = "latest model manifest and prepared ORT session disagree";
        return false;
    }
    if (config_.rollout.tmax == 0) {
        error = "AIServer rollout tmax is invalid";
        return false;
    }

    std::ostringstream segment_id;
    segment_id << producer_instance_id_ << '/' << session.session_id << '/'
               << session.current_episode_id << "/agent-" << agent_id
               << "/segment-" << next_segment_sequence;
    ++next_segment_sequence;
    if (next_segment_sequence == 0) ++next_segment_sequence;
    agent.segment_open = true;
    agent.segment_id = segment_id.str();
    agent.pinned_model = model_manifest_;
    agent.pinned_prepared_model = std::move(prepared);
    agent.segment_transitions.clear();
    ++per_agent_activation_count;
    latest_model_used = true;
    LOG_INFO("TrainingRuntime",
             "Agent segment 激活: session=%s episode=%s agent_id=%d segment=%s model_step=%llu",
             session.session_id.c_str(), session.current_episode_id.c_str(),
             agent_id, agent.segment_id.c_str(),
             static_cast<unsigned long long>(
                 agent.pinned_model.model_step()));
    error.clear();
    return true;
}

template <class Sessions>
bool TrainingRuntime<Sessions>::InferPinnedValue(
    const AgentRuntime& agent,
    const std::vector<float>& observation,
    float& value) {
    if (!agent.segment_open || !agent.pinned_prepared_model.valid() ||
        static_cast<int>(observation.size()) !=
            config_.model.expected_obs_dim) {
        MarkDegraded("pinned bootstrap inference inputs are incomplete");
        return false;
    }
    std::vector<float> logits;
    std::string output_error;
    const auto start = std::chrono::steady_clock::now();
    const bool inferred = onnx_inferencer_.InferPrepared(
        agent.pinned_prepared_model, observation,
        static_cast<int>(observation.size()), logits, value, &output_error);
    const double latency_ms = training_runtime_detail::ElapsedMs(start);
    ++inference_count_;
    inference_latency_sum_ms_ += latency_ms;
    inference_latency_max_ms_ =
        std::max(inference_latency_max_ms_, latency_ms);
    if (!inferred || !ValidateEpisodeModelOutput(
                         logits, config_.model.expected_action_dim, value,
                         output_error)) {
        MarkDegraded("pinned bootstrap inference failed: " + output_error);
        return false;
    }
    return true;
}

template <class Sessions>
bool TrainingRuntime<Sessions>::ActivateStagedModel() {
    if (!staged_model_manifest_.HasModelIdentity() ||
        !staged_prepared_model_.valid()) {
        return true;
    }

    std::string error;
    if (!ValidateStagedModelProgress(
            model_manifest_, staged_model_manifest_, error)) {
        MarkDegraded("staged model progress is invalid: " + error);
        return false;
    }

    common::ServiceInstanceIdentity ack_authority;
    const auto authority_probe =
        model_distributor_.ProbeAckAuthorityDisposition(
            ack_authority, error);
    if (authority_probe ==
        ModelDistributorClient::AuthorityProbeDisposition::Retryable) {
        LOG_ERROR("TrainingRuntime", "模型激活延后: %s", error.c_str());
        return false;
    }
    if (authority_probe ==
        ModelDistributorClient::AuthorityProbeDisposition::Rejected) {
        model_state_.store(training::MODEL_STATE_FAILED);
        MarkDegraded("model ACK authority probe was rejected: " + error);
        RecordModelFeedback(staged_model_manifest_.wire.identity(),
                            "ack_rejected", last_error_);
        return false;
    }
    ModelManifest candidate = staged_model_manifest_;

    const auto ack_deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(config_.model.startup_timeout_ms);
    std::string ack_error;
    const auto ack = model_distributor_.AckIdempotently(
        candidate, config_.sample_distributor.aiserver_id,
        training::MODEL_LOAD_STATUS_LOADED, "loaded", ack_error,
        &ack_authority, ack_deadline);
    if (ack == ModelDistributorClient::AckDisposition::Rejected ||
        ack == ModelDistributorClient::AckDisposition::NotApplied) {
        model_state_.store(training::MODEL_STATE_FAILED);
        state_.store(training::AISERVER_STATE_DEGRADED);
        last_error_ = "model ACK was rejected: " + ack_error;
        RecordModelFeedback(candidate.wire.identity(), "ack_rejected", last_error_);
        LOG_ERROR("TrainingRuntime", "%s", last_error_.c_str());
        return false;
    }

    if (ack == ModelDistributorClient::AckDisposition::Uncertain) {
        RecordPendingModelAck(candidate, ack_authority, ack_deadline, ack_error);
        return false;
    }
    if (model_manifest_.HasModelIdentity() &&
        !latest_prepared_used_by_agent_) {
        ++superseded_without_agent_activation_count_;
    }
    staged_prepared_model_.model_path = candidate.model_path;
    onnx_inferencer_.ActivatePreparedModel(
        std::move(staged_prepared_model_));
    model_manifest_ = candidate;
    staged_model_manifest_ = ModelManifest{};
    staged_prepared_model_ = OnnxInferencer::PreparedModel{};
    ++model_switch_count_;
    latest_prepared_used_by_agent_ = false;
    model_state_.store(training::MODEL_STATE_READY);
    LOG_INFO(
        "TrainingRuntime", "模型已成为 latest-prepared: lineage=%s model_step=%llu",
        model_manifest_.model_lineage_id().c_str(),
        static_cast<unsigned long long>(model_manifest_.model_step()));
    return true;
}

template <class Sessions>
bool TrainingRuntime<Sessions>::PrepareModelAction(
    Session& session,
    AgentRuntime& agent,
    int agent_id,
    std::vector<float> obs,
    int64_t action_frame_id,
    const std::vector<bool>& action_mask,
    uint64_t& next_segment_sequence,
    int64_t& per_agent_activation_count,
    bool& latest_model_used,
    std::mt19937& action_rng,
    int& action,
    float& log_prob,
    float& value) {
    const bool training_episode =
        session.current_episode_mode == PolicyMode::Training;
    std::string pin_error;
    if (training_episode &&
        !EnsureAgentSegmentPin(
            session, agent, agent_id, next_segment_sequence,
            per_agent_activation_count, latest_model_used, pin_error)) {
        MarkDegraded("Agent segment pin failed: " + pin_error);
        return false;
    }

    std::vector<float> logits;
    std::string inference_error;
    const auto start = std::chrono::steady_clock::now();
    const bool inferred = training_episode
        ? onnx_inferencer_.InferPrepared(
              agent.pinned_prepared_model, obs,
              static_cast<int>(obs.size()), logits, value, &inference_error)
        : onnx_inferencer_.Infer(
              obs, static_cast<int>(obs.size()), logits, value, &inference_error);
    const double latency_ms = training_runtime_detail::ElapsedMs(start);
    ++inference_count_;
    inference_latency_sum_ms_ += latency_ms;
    inference_latency_max_ms_ =
        std::max(inference_latency_max_ms_, latency_ms);

    if (!inferred) {
        MarkDegraded("ONNX inference failed: " + inference_error);
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
            logits, action_mask, session.current_episode_mode,
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
    agent.pending_action_mask = action_mask;
    agent.pending_action = action;
    agent.pending_action_frame_id = action_frame_id;
    agent.pending_log_prob = log_prob;
    agent.pending_value = value;
    agent.has_pending_action = true;
    return true;
}

template <class Sessions>
bool TrainingRuntime<Sessions>::FinalizePendingTransition(
    Session& session,
    int agent_id,
    const RewardResult& reward,
    const std::vector<float>& next_observation,
    bool collect_training_sample,
    std::string& error) {
    auto agent_it = session.agents.find(agent_id);
    if (agent_it == session.agents.end()) {
        error = "transition Agent identity is unknown";
        return false;
    }
    auto& agent = agent_it->second;
    if (!agent.has_pending_action) return true;
    RawRolloutTransition transition;
    if (collect_training_sample) {
        if (!agent.segment_open ||
            !agent.pinned_model.HasModelIdentity() ||
            !agent.pinned_prepared_model.valid()) {
            error = "training action result has no pinned Agent segment";
            return false;
        }
        if (!reward.valid) {
            error = reward.error;
            return false;
        }
        if (!BuildRawRolloutTransition(
                agent, next_observation, reward,
                config_.model.expected_obs_dim,
                config_.model.expected_action_dim,
                config_.policy.action_mask_mode,
                transition, error)) {
            error = "training transition continuity is invalid: " + error;
            return false;
        }
    }

    // Commit only after all workload-specific validation has succeeded.
    // Reward and pinned-model transitions belong exclusively to the training
    // workload. Standalone evaluation advances inference/lifecycle state only.
    if (collect_training_sample) {
        if (agent.pinned_model.model_lineage_id().empty()) {
            error = "training transition behavior model identity is incomplete";
            return false;
        }
        const ModelStep behavior_step = agent.pinned_model.model_step();
        if (!agent.episode_behavior_model_seen) {
            agent.episode_behavior_model_seen = true;
            agent.minimum_episode_behavior_model_step = behavior_step;
            agent.maximum_episode_behavior_model_step = behavior_step;
            agent.episode_behavior_model_lineage_id =
                agent.pinned_model.model_lineage_id();
        } else {
            if (agent.episode_behavior_model_lineage_id !=
                agent.pinned_model.model_lineage_id()) {
                error = "training Episode contains mixed model lineages";
                return false;
            }
            agent.minimum_episode_behavior_model_step = std::min(
                agent.minimum_episode_behavior_model_step, behavior_step);
            agent.maximum_episode_behavior_model_step = std::max(
                agent.maximum_episode_behavior_model_step, behavior_step);
        }
    }
    ++agent.episode_transition_count;
    if (collect_training_sample) {
        session.reward_metrics.Observe(reward.total, reward.items);
        agent.episode_return += reward.total;
        for (const auto& item : reward.items) {
            agent.reward_component_sums[item.first] += item.second;
        }
    }
    if (collect_training_sample) {
        agent.last_completed_transition_at_unix_ms =
            transition.created_at_unix_ms;
        agent.segment_transitions.push_back(std::move(transition));
    }
    agent.has_pending_action = false;
    agent.pending_action_frame_id = -1;
    agent.pending_obs.clear();
    agent.pending_action_mask.clear();

    error.clear();
    return true;
}

template <class Sessions>
bool TrainingRuntime<Sessions>::PrepareAgentSegmentClose(
    Session& session,
    int agent_id,
    training::SegmentCloseReason reason,
    float bootstrap_value,
    bool bootstrap_applied,
    int64_t& produced_unique_transitions,
    int64_t& produced_unique_envelopes,
    std::unordered_map<ModelStep, int64_t>& produced_by_model,
    std::vector<training::ProcessedTransitionEnvelope>& envelopes,
    std::unordered_map<training::SegmentCloseReason, int64_t>& close_counts,
    std::string& error) {
    auto agent_it = session.agents.find(agent_id);
    if (agent_it == session.agents.end()) {
        error = "segment close Agent identity is unknown";
        return false;
    }
    auto& agent = agent_it->second;
    if (!agent.segment_open || agent.segment_transitions.empty() ||
        agent.segment_id.empty() ||
        !agent.pinned_model.HasModelIdentity() ||
        !agent.pinned_prepared_model.valid()) {
        error = "cannot close an empty or unpinned Agent segment";
        return false;
    }
    if (reason == training::SEGMENT_CLOSE_REASON_UNSPECIFIED) {
        error = "segment close reason must be explicit";
        return false;
    }
    const bool terminal =
        reason == training::SEGMENT_CLOSE_REASON_ENVIRONMENT_TERMINATED;
    if ((terminal && (bootstrap_applied || bootstrap_value != 0.0f)) ||
        (!terminal && !bootstrap_applied) ||
        !std::isfinite(bootstrap_value)) {
        error = "segment close bootstrap facts contradict the close reason";
        return false;
    }

    std::vector<float> advantages;
    std::vector<float> value_targets;
    if (!EstimateRolloutSegment(
            agent.segment_transitions, config_.rollout.gamma,
            config_.rollout.gae_lambda, bootstrap_value,
            advantages, value_targets, error)) {
        return false;
    }

    const uint32_t segment_size = static_cast<uint32_t>(
        agent.segment_transitions.size());
    std::vector<training::ProcessedTransition> processed;
    if (!ProjectProcessedSegment(
            agent.segment_transitions, advantages, value_targets,
            agent.segment_id, agent.pinned_model.wire.identity(),
            config_.model.expected_obs_dim,
            config_.model.expected_action_dim,
            config_.policy.action_mask_mode,
            processed, error)) {
        return false;
    }

    std::size_t offset = 0;
    std::size_t chunk_index = 0;
    std::vector<training::ProcessedTransitionEnvelope> built;
    while (offset < processed.size()) {
        training::ProcessedTransitionEnvelope envelope;
        envelope.set_envelope_id(
            agent.segment_id + "/envelope-" +
            std::to_string(chunk_index));
        training_runtime_detail::FillServiceIdentity(
            "sample-distributor", producer_instance_id_,
            producer_lifecycle_epoch_, envelope.mutable_producer());
        envelope.mutable_behavior_model()->CopyFrom(
            agent.pinned_model.wire.identity());

        while (offset < processed.size() &&
               envelope.samples_size() <
                   config_.sample_distributor.envelope_max_transitions) {
            *envelope.add_samples() = processed[offset];
            if (envelope.ByteSizeLong() >
                config_.sample_distributor.envelope_max_bytes) {
                envelope.mutable_samples()->RemoveLast();
                break;
            }
            ++offset;
        }
        if (envelope.samples_size() == 0) {
            error = "one processed sample exceeds envelope_max_bytes";
            return false;
        }
        if (envelope.ByteSizeLong() >
            config_.sample_distributor.envelope_max_bytes) {
            error = "processed envelope exceeds envelope_max_bytes";
            return false;
        }
        built.push_back(std::move(envelope));
        ++chunk_index;
    }

    produced_unique_transitions +=
        static_cast<int64_t>(agent.segment_transitions.size());
    produced_unique_envelopes += static_cast<int64_t>(built.size());
    produced_by_model[agent.pinned_model.model_step()] +=
        static_cast<int64_t>(agent.segment_transitions.size());
    envelopes.insert(envelopes.end(),
                     std::make_move_iterator(built.begin()),
                     std::make_move_iterator(built.end()));
    ++close_counts[reason];
    LOG_INFO(
        "TrainingRuntime",
        "Agent segment 封口: session=%s episode=%s agent_id=%d segment=%s model_step=%llu reason=%s transitions=%u envelopes=%zu bootstrap_applied=%d bootstrap_value=%.9g",
        session.session_id.c_str(), session.current_episode_id.c_str(),
        agent_id, agent.segment_id.c_str(),
        static_cast<unsigned long long>(agent.pinned_model.model_step()),
        training::SegmentCloseReason_Name(reason).c_str(), segment_size,
        built.size(), bootstrap_applied ? 1 : 0,
        static_cast<double>(bootstrap_value));
    agent.segment_open = false;
    agent.segment_id.clear();
    agent.pinned_model = ModelManifest{};
    agent.pinned_prepared_model = OnnxInferencer::PreparedModel{};
    agent.segment_transitions.clear();
    error.clear();
    return true;
}

template <class Sessions>
void TrainingRuntime<Sessions>::DiscardAgentSegment(
    Session& session,
    int agent_id,
    training::SegmentCloseReason reason,
    bool exclude_pending_action,
    int64_t& quarantined_transitions,
    int64_t& pending_actions_excluded,
    int64_t& closed_segments,
    std::unordered_map<training::SegmentCloseReason, int64_t>& close_counts) {
    auto agent_it = session.agents.find(agent_id);
    if (agent_it == session.agents.end()) return;
    auto& agent = agent_it->second;
    if (exclude_pending_action && agent.has_pending_action) {
        ++pending_actions_excluded;
    }
    if (agent.segment_open || agent.has_pending_action ||
        !agent.segment_transitions.empty()) {
        quarantined_transitions +=
            static_cast<int64_t>(agent.segment_transitions.size());
        ++close_counts[reason];
        ++closed_segments;
    }
    agent.has_pending_action = false;
    agent.pending_action_frame_id = -1;
    agent.pending_obs.clear();
    agent.pending_action_mask.clear();
    agent.segment_open = false;
    agent.segment_id.clear();
    agent.pinned_model = ModelManifest{};
    agent.pinned_prepared_model = OnnxInferencer::PreparedModel{};
    agent.segment_transitions.clear();
}

template <class Sessions>
training::GetMetricCatalogRsp TrainingRuntime<Sessions>::MetricCatalog() const {
    auto catalog = metric_registry_.Catalog(MetricSourceIdentity());
    const std::string method = "rl.training.v1.AIServerTrainingStatusService/GetAIServerStatus";
    using namespace rl::training::v1;
    rl_sdk::AddStatusMetric(catalog, "sample.flow.produced.total", "Produced Samples", "sample_flow", "count", "aiserver", method, "produced_unique_transitions");
    rl_sdk::AddStatusMetric(catalog, "sample.flow.outbound_pending.total", "Outbound Pending", "sample_flow", "count", "aiserver", method, "outbound_queue_transitions");
    rl_sdk::AddStatusMetric(catalog, "sample.flow.final_drop.total", "Final Drop", "sample_flow", "count", "aiserver", method, "final_drop_unique_transitions");
    rl_sdk::AddStatusMetric(catalog, "server.latency.sample_send.mean_ms", "Sample Send Latency", "latency", "ms", "push_rpc", method, "push_rpc_latency_sum_ms", METRIC_VALUE_TYPE_SUM_COUNT, "push_rpc_count");
    rl_sdk::AddStatusMetric(catalog, "server.latency.inference.mean_ms", "Inference Latency Mean", "latency", "ms", "inference", method, "inference_latency_sum_ms", METRIC_VALUE_TYPE_SUM_COUNT, "inference_count");
    rl_sdk::AddStatusMetric(catalog, "server.latency.inference.max_ms", "Inference Latency Max", "latency", "ms", "inference", method, "inference_latency_max_ms", METRIC_VALUE_TYPE_SCALAR, "inference_count");
    rl_sdk::AddStatusMetric(catalog, "server.latency.update_rpc.mean_ms", "Update RPC Latency Mean", "latency", "ms", "update_rpc", method, "update_rpc_latency_sum_ms", METRIC_VALUE_TYPE_SUM_COUNT, "update_rpc_count");
    rl_sdk::AddStatusMetric(catalog, "server.latency.update_rpc.max_ms", "Update RPC Latency Max", "latency", "ms", "update_rpc", method, "update_rpc_latency_max_ms", METRIC_VALUE_TYPE_SCALAR, "update_rpc_count");
    return catalog;
}

template <class Sessions>
bool TrainingRuntime<Sessions>::FlushRewardMetricsLocked() {
    const auto ended_at = std::chrono::steady_clock::now();
    const int64_t end = NowMs();
    for (const auto& id : session_mgr_.GetSessionIds()) {
        auto* session = session_mgr_.GetSession(id);
        if (!session || !session->reward_metrics.count) continue;
        training::RegisteredMetricRecord record;
        (*record.mutable_attributes())["environment_instance_id"] = session->environment_instance_id;
        (*record.mutable_attributes())["episode_id"] = session->current_episode_id;
        session->reward_metrics.AppendTo(record, metric_registry_, reward_metric_prefix_, end, ended_at);
        std::string payload;
        if (!record.SerializeToString(&payload) || !metric_events_.AppendFact(std::move(payload), end).applied()) {
            LOG_ERROR("MetricEvent", "reward interval could not enter local journal: session=%s", id.c_str());
            return false; // Stop periodic output and retain the unaccepted increment.
        }
        session->reward_metrics = {};
    }
    return true;
}

template <class Sessions>
void TrainingRuntime<Sessions>::RecordUpdateLatency(
    std::chrono::steady_clock::time_point start) {
    const double latency_ms = training_runtime_detail::ElapsedMs(start);
    ++update_rpc_count_;
    update_rpc_latency_sum_ms_ += latency_ms;
    update_rpc_latency_max_ms_ =
        std::max(update_rpc_latency_max_ms_, latency_ms);
}

template <class Sessions>
void TrainingRuntime<Sessions>::MarkDegraded(const std::string& error) {
    state_.store(training::AISERVER_STATE_DEGRADED);
    last_error_ = error;
    LOG_ERROR("TrainingRuntime", "进入 DEGRADED: %s", error.c_str());
}

template <class Sessions>
int64_t TrainingRuntime<Sessions>::CountCachedTransitions() {
    int64_t count = 0;
    for (const auto& session_id : session_mgr_.GetSessionIds()) {
        const auto* session = session_mgr_.GetSession(session_id);
        if (!session) continue;
        for (const auto& item : session->agents) {
            count += static_cast<int64_t>(
                item.second.segment_transitions.size());
        }
    }
    return count;
}

template <class Sessions>
int64_t TrainingRuntime<Sessions>::CountCachedSegments() {
    int64_t count = 0;
    for (const auto& session_id : session_mgr_.GetSessionIds()) {
        const auto* session = session_mgr_.GetSession(session_id);
        if (!session) continue;
        for (const auto& item : session->agents) {
            if (item.second.segment_open) ++count;
        }
    }
    return count;
}

template <class Sessions>
grpc::Status TrainingRuntime<Sessions>::GetAIServerStatus(
    grpc::ServerContext*,
    const training::AIServerStatusReq*,
    training::AIServerStatusRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto sender = sample_distributor_.GetSnapshot();
    const int64_t timestamp = NowMs();
    training_runtime_detail::FillServiceIdentity("rl-aiserver", producer_instance_id_,
                        producer_lifecycle_epoch_,
                        rsp->mutable_aiserver());
    rsp->set_state(state_.load());
    rsp->set_ready(IsReady());
    rsp->set_distributor_ready(
        config_.server.run_mode != aiserver_mode::kTraining ||
        (sender.ready && !sender.transient_retry &&
         !sender.terminal_fault));
    rsp->set_model_state(model_state_.load());
    {
        std::lock_guard<std::mutex> feedback_lock(model_feedback_mutex_);
        *rsp->mutable_model_feedback() = model_feedback_;
    }
    if (model_manifest_.HasModelIdentity()) {
        rsp->mutable_loaded_model()->CopyFrom(
            model_manifest_.wire.identity());
    }
    if (staged_model_manifest_.HasModelIdentity()) {
        rsp->mutable_staged_model()->CopyFrom(
            staged_model_manifest_.wire.identity());
    }

    rsp->set_outbound_queue_envelopes(
        static_cast<int64_t>(sender.queue_envelopes));
    rsp->set_outbound_queue_transitions(
        sender.queue_transitions);
    rsp->set_outbound_queue_estimated_bytes(
        sender.queue_estimated_bytes);
    rsp->set_outbound_queue_high_watermark(
        sender.queue_high_watermark);
    rsp->set_produced_unique_transitions(
        produced_unique_transitions_);
    rsp->set_produced_unique_envelopes(
        produced_unique_envelopes_);
    rsp->set_push_attempt_count(sender.push_attempt_count);
    rsp->set_accepted_unique_transitions(
        sender.accepted_unique_transitions);
    rsp->set_duplicate_push_attempt_count(
        sender.duplicate_push_attempt_count);
    rsp->set_rejected_push_attempt_count(
        sender.rejected_push_attempt_count);
    rsp->set_retry_attempt_count(sender.retry_attempt_count);
    rsp->set_final_drop_unique_transitions(
        sender.final_drop_unique_transitions);

    const auto client_activity =
        session_mgr_.GetClientActivitySnapshot(
            timestamp,
            config_.sample_distributor.recovery_timeout_ms);
    const int active_session_count =
        client_activity.active_session_count;
    rsp->set_active_actor_session_count(active_session_count);
    rsp->set_active_segment_count(CountCachedSegments());
    rsp->set_update_rpc_count(update_rpc_count_);
    rsp->set_update_rpc_latency_sum_ms(update_rpc_latency_sum_ms_);
    rsp->set_update_rpc_latency_max_ms(update_rpc_latency_max_ms_);
    rsp->set_inference_count(inference_count_);
    rsp->set_inference_latency_sum_ms(
        inference_latency_sum_ms_);
    rsp->set_inference_latency_max_ms(
        inference_latency_max_ms_);
    rsp->set_push_rpc_count(sender.push_rpc_count);
    rsp->set_push_rpc_latency_sum_ms(
        sender.push_rpc_latency_sum_ms);
    rsp->set_push_rpc_latency_max_ms(
        sender.push_rpc_latency_max_ms);
    rsp->set_model_switch_count(model_switch_count_);
    rsp->set_quarantined_transition_count(
        quarantined_transition_count_);
    rsp->set_quarantined_envelope_count(
        quarantined_envelope_count_);
    rsp->set_closed_segment_count(closed_segment_count_);
    rsp->set_pending_action_excluded_count(
        pending_action_excluded_count_);
    rsp->set_rollout_estimator_failure_count(
        rollout_estimator_failure_count_);
    rsp->set_per_agent_model_activation_count(
        per_agent_model_activation_count_);
    rsp->set_superseded_without_agent_activation_count(
        superseded_without_agent_activation_count_);

    std::vector<std::pair<training::SegmentCloseReason, int64_t>>
        close_counts(segment_close_counts_.begin(),
                     segment_close_counts_.end());
    std::sort(
        close_counts.begin(), close_counts.end(),
        [](const auto& lhs, const auto& rhs) {
            return static_cast<int>(lhs.first) <
                   static_cast<int>(rhs.first);
        });
    for (const auto& item : close_counts) {
        auto* target = rsp->add_segment_close_counts();
        target->set_reason(item.first);
        target->set_count(item.second);
    }

    rsp->set_last_error(
        last_error_.empty() ? sender.last_error : last_error_);
    rsp->set_timestamp_unix_ms(timestamp);
    return grpc::Status::OK;
}
