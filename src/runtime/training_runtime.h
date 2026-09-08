#pragma once

#include "ai/onnx_inferencer.h"
#include "config/training_runtime_config.h"
#include "contracts/training_namespaces.h"
#include "proto/training/training.grpc.pb.h"
#include "metrics/metric_event_service.h"
#include "metrics/periodic_flush.h"
#include "model/model_manifest.h"
#include "model/model_distributor_client.h"
#include "sample/sample_sender.h"
#include "sample/reward_result.h"
#include "session/training_session.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <unordered_map>

// The typed RPC adapter serializes command preparation/commit with mutex_.
// Session candidates include task state, but this runtime only accesses the
// common training fields. Model watching and sample delivery share this owner.
template <class Sessions>
class TrainingRuntime {
public:
    using Session = typename Sessions::Session;
    using AgentRuntime = typename Sessions::AgentRuntime;

    TrainingRuntime(const TrainingRuntimeConfig& config, std::string reward_metric_prefix);
    ~TrainingRuntime();
    bool Start();
    bool BeginShutdown();
    bool IsReady() const;
    common::ServiceInstanceIdentity MetricSourceIdentity() const;
    training::GetMetricCatalogRsp MetricCatalog() const;
    grpc::Status GetAIServerStatus(grpc::ServerContext*, const training::AIServerStatusReq*, training::AIServerStatusRsp*);

    bool LoadInitialModel();
    bool AcquireTrainingWorkspaceLease(std::string& error);
    bool ReleaseTrainingWorkspaceLease(std::string& error);
    bool PrepareModelArtifact(
        const ModelManifest& manifest,
        OnnxInferencer::PreparedModel& prepared,
        std::string& error);
    bool FetchPrepareAndPublishModel(
        ModelStep model_step,
        const std::string& lineage_id,
        ModelManifest& manifest,
        OnnxInferencer::PreparedModel& prepared,
        std::string& error);
    std::set<ModelStep> ProtectedCachedModelStepsLocked();
    bool IsCoreInferenceReady() const;
    void StartModelWatcher();
    void StopModelWatcher();
    void ModelWatchLoop();
    void RecordModelFeedback(const training::ModelIdentity& candidate,
                             const std::string& stage, const std::string& error);
    // Watcher I/O runs outside mutex_; status reads take only this short lock.
    std::mutex model_feedback_mutex_;
    training::ModelFeedbackStatus model_feedback_;
    void RecordPendingModelAck(
        const ModelManifest& manifest,
        const common::ServiceInstanceIdentity& authority,
        const std::string& error);
    bool RetryPendingModelAck();
    bool TryPromoteStagedModelForWatcher();
    bool RecoverExpiredClientSessions();
    bool ActivateStagedModel();
    bool ValidateStagedModelProgress(const ModelManifest& active,
                                     const ModelManifest& candidate,
                                     std::string& error) const;
    bool FinalizePendingTransition(Session& session,
                                   int agent_id,
                                   const RewardResult& reward,
                                   const std::vector<float>& next_observation,
                                   bool collect_training_sample,
                                   std::string& error);
    bool InferPinnedValue(
        const AgentRuntime& agent,
        const std::vector<float>& observation,
        float& value);
    bool EnsureAgentSegmentPin(
        Session& session,
        AgentRuntime& agent,
        int agent_id,
        uint64_t& next_segment_sequence,
        int64_t& per_agent_activation_count,
        bool& latest_model_used,
        std::string& error);
    bool PrepareModelAction(
        Session& session,
        AgentRuntime& agent,
        int agent_id,
        std::vector<float> observation,
        int64_t action_frame_id,
        const std::vector<bool>& action_mask,
        uint64_t& next_segment_sequence,
        int64_t& per_agent_activation_count,
        bool& latest_model_used,
        std::mt19937& action_rng,
        int& action,
        float& log_prob,
        float& value);
    bool PrepareAgentSegmentClose(
        Session& session,
        int agent_id,
        training::SegmentCloseReason reason,
        float bootstrap_value,
        bool bootstrap_applied,
        int64_t& produced_unique_transitions,
        int64_t& produced_unique_envelopes,
        std::unordered_map<ModelStep, int64_t>& produced_by_model,
        std::vector<training::ProcessedTransitionEnvelope>& envelopes,
        std::unordered_map<training::SegmentCloseReason, int64_t>&
            close_counts,
        std::string& error);
    void DiscardAgentSegment(Session& session,
                             int agent_id,
                             training::SegmentCloseReason reason,
                             bool exclude_pending_action,
                             int64_t& quarantined_transitions,
                             int64_t& pending_actions_excluded,
                             int64_t& closed_segments,
                             std::unordered_map<
                                 training::SegmentCloseReason,
                                 int64_t>& close_counts);
    int64_t CountCachedTransitions();
    int64_t CountCachedSegments();
    void RecordUpdateLatency(std::chrono::steady_clock::time_point start);
    void MarkDegraded(const std::string& error);
    static int64_t NowMs();
    static std::string CreateProducerInstanceId(const std::string& aiserver_id);
    static uint64_t CreateProducerLifecycleEpoch();

    const TrainingRuntimeConfig config_;
    const std::string reward_metric_prefix_;
    Sessions session_mgr_;
    mutable std::mutex mutex_;
    std::string producer_instance_id_;
    uint64_t producer_lifecycle_epoch_ = 0;
    SampleDistributor sample_distributor_;
    OnnxInferencer onnx_inferencer_;
    ModelDistributorClient model_distributor_;
    ModelManifest model_manifest_;
    ModelManifest staged_model_manifest_;
    OnnxInferencer::PreparedModel staged_prepared_model_;
    bool model_ack_pending_ = false;
    ModelManifest pending_model_ack_manifest_;
    common::ServiceInstanceIdentity pending_model_ack_authority_;
    std::string pending_model_ack_error_;
    std::string pending_model_ack_cause_;
    std::string training_workspace_lock_path_;
    bool training_workspace_lease_held_ = false;

    std::atomic<training::AIServerState> state_{training::AISERVER_STATE_STARTING};
    std::atomic<training::ModelState> model_state_{training::MODEL_STATE_WAITING};
    std::atomic<uint64_t> next_segment_seq_{1};
    std::atomic<uint64_t> next_episode_id_{1};
    std::atomic<uint64_t> next_lifecycle_epoch_{1};
    std::atomic<bool> model_watch_stop_{false};
    std::thread model_watch_thread_;
    std::mt19937 action_rng_;
    std::unordered_map<std::string, std::string> open_payloads_;
    std::unordered_map<std::string, std::string> open_responses_;
    MetricEventJournal metric_events_;
    MetricRegistry metric_registry_;
    MetricEventService metric_service_{metric_events_};
    PeriodicMetricFlush metric_flush_;
    bool FlushRewardMetricsLocked();
    bool started_ = false;
    bool shutdown_started_ = false;
    bool shutdown_completed_ = false;
    bool shutdown_succeeded_ = true;
    bool client_initialized_ = false;

    int64_t produced_unique_transitions_ = 0;
    std::unordered_map<ModelStep, int64_t> produced_transitions_by_model_;
    int64_t produced_unique_envelopes_ = 0;
    int64_t enqueue_count_ = 0;
    double enqueue_latency_sum_ms_ = 0.0;
    double enqueue_latency_max_ms_ = 0.0;
    int64_t update_rpc_count_ = 0;
    double update_rpc_latency_sum_ms_ = 0.0;
    double update_rpc_latency_max_ms_ = 0.0;
    int64_t inference_count_ = 0;
    double inference_latency_sum_ms_ = 0.0;
    double inference_latency_max_ms_ = 0.0;
    int64_t model_switch_count_ = 0;
    int64_t quarantined_transition_count_ = 0;
    int64_t quarantined_envelope_count_ = 0;
    int64_t closed_segment_count_ = 0;
    int64_t pending_action_excluded_count_ = 0;
    int64_t rollout_estimator_failure_count_ = 0;
    int64_t per_agent_model_activation_count_ = 0;
    int64_t superseded_without_agent_activation_count_ = 0;
    bool latest_prepared_used_by_agent_ = false;
    std::unordered_map<training::SegmentCloseReason, int64_t>
        segment_close_counts_;
    std::string last_error_;
};

#include "runtime/training_runtime.tpp"
