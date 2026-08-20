#pragma once

#include "ai/astar_solver.h"
#include "ai/maze_reward.h"
#include "ai/onnx_inferencer.h"
#include "config/config_loader.h"
#include "contracts/contract_namespaces.h"
#include "maze_task.grpc.pb.h"
#include "training.grpc.pb.h"
#include "metrics/episode_metrics.h"
#include "model/model_manifest.h"
#include "model/model_distributor_client.h"
#include "sample/sample_sender.h"
#include "session/session_manager.h"
#include "task/single_map_task_controller.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class MazeServiceImpl final : public maze::MazeTaskService::Service,
                              public training::AIServerTrainingStatusService::Service,
                              public training::MetricEventService::Service {
public:
    explicit MazeServiceImpl(const AIServerConfig& config);
    ~MazeServiceImpl();

    bool Start();
    bool BeginShutdown();
    bool IsReady() const;

    grpc::Status OpenSession(grpc::ServerContext* ctx,
                             const maze::OpenSessionReq* req,
                             maze::OpenSessionRsp* rsp) override;

    grpc::Status Init(grpc::ServerContext* ctx,
                      const maze::InitReq* req,
                      maze::InitRsp* rsp) override;

    grpc::Status BeginEpisode(grpc::ServerContext* ctx,
                              const maze::BeginEpisodeReq* req,
                              maze::BeginEpisodeRsp* rsp) override;

    grpc::Status Update(grpc::ServerContext* ctx,
                        const maze::UpdateReq* req,
                        maze::UpdateRsp* rsp) override;

    grpc::Status EndEpisode(grpc::ServerContext* ctx,
                            const maze::EndEpisodeReq* req,
                            maze::EndEpisodeRsp* rsp) override;

    grpc::Status AbortEpisode(grpc::ServerContext* ctx,
                              const maze::AbortEpisodeReq* req,
                              maze::AbortEpisodeRsp* rsp) override;

    grpc::Status CloseSession(grpc::ServerContext* ctx,
                              const maze::CloseSessionReq* req,
                              maze::CloseSessionRsp* rsp) override;

    grpc::Status GetAIServerStatus(grpc::ServerContext* ctx,
                                   const training::AIServerStatusReq* req,
                                   training::AIServerStatusRsp* rsp) override;

    grpc::Status GetMetricBatch(
        grpc::ServerContext* ctx,
        const training::GetMetricBatchReq* req,
        training::GetMetricBatchRsp* rsp) override;
    grpc::Status AckMetricBatch(
        grpc::ServerContext* ctx,
        const training::AckMetricBatchReq* req,
        training::AckMetricBatchRsp* rsp) override;

private:
    friend struct MazeServiceUpdateTestAccess;
    friend struct MazeServiceLifecycleTestAccess;

    bool LoadInitialModel();
    bool AcquireTrainingWorkspaceLease(std::string& error);
    bool ReleaseTrainingWorkspaceLease(std::string& error);
    bool LoadAndPrepareCachedModel(
        const ModelManifest& manifest,
        OnnxInferencer::PreparedModel& prepared,
        std::string& error);
    bool FetchPrepareAndPublishModel(
        ModelStep model_step,
        ModelManifest& manifest,
        OnnxInferencer::PreparedModel& prepared,
        std::string& error,
        bool force_exact_download = false);
    bool BackfillOneCachedModel(
        const ModelDistributorClient::AvailableRange& range,
        std::string& error);
    std::set<ModelStep> ProtectedCachedModelStepsLocked();
    bool IsCoreInferenceReady() const;
    void StartModelWatcher();
    void StopModelWatcher();
    void ModelWatchLoop();
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
    void InitAgentSolver(SessionManager::AgentRuntime& agent,
                         const SessionManager::Session& session);
    training::EpisodeMetricFact BuildEpisodeMetricFact(
        const SessionManager::Session& session,
        const std::vector<AgentEpisodeResult>& agents) const;
    maze::EpisodeOutcome BuildEpisodeOutcome(
        const SessionManager::Session& session,
        const std::vector<AgentEpisodeResult>& agents) const;
    void ResetEpisodeState(SessionManager::Session& session,
                           const std::string& episode_id);
    bool FinalizePendingTransition(SessionManager::Session& session,
                                   int agent_id,
                                   int gx,
                                   int gy,
                                   bool is_done,
                                   maze::MazeTerminationReason reason,
                                   bool collect_training_sample,
                                   std::string& error);
    bool InferPinnedValue(
        const SessionManager::AgentRuntime& agent,
        const std::vector<float>& observation,
        float& value);
    bool EnsureAgentSegmentPin(
        SessionManager::Session& session,
        SessionManager::AgentRuntime& agent,
        int agent_id,
        uint64_t& next_segment_sequence,
        int64_t& per_agent_activation_count,
        bool& latest_model_used,
        std::string& error);
    bool PrepareModelAction(
        SessionManager::Session& session,
        SessionManager::AgentRuntime& agent,
        int agent_id,
        int gx,
        int gy,
        int64_t action_frame_id,
        uint64_t& next_segment_sequence,
        int64_t& per_agent_activation_count,
        bool& latest_model_used,
        std::mt19937& action_rng,
        int& action,
        float& log_prob,
        float& value);
    bool PrepareAgentSegmentClose(
        SessionManager::Session& session,
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
    void DiscardAgentSegment(SessionManager::Session& session,
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
    SingleMapModelIdentity ActiveModelIdentity() const;
    bool WriteTaskControllerReceipt(
        const SingleMapTaskController& controller,
        std::string& error) const;
    static int64_t NowMs();
    static std::string CreateProducerInstanceId(const std::string& aiserver_id);
    static uint64_t CreateProducerLifecycleEpoch();

    AIServerConfig config_;
    SessionManager session_mgr_;
    mutable std::mutex mutex_;
    std::string producer_instance_id_;
    uint64_t producer_lifecycle_epoch_ = 0;
    SampleDistributor sample_distributor_;
    EpisodeMetricsWindow episode_metrics_;
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
    std::atomic<uint64_t> next_metric_sequence_{1};
    std::atomic<bool> model_watch_stop_{false};
    std::thread model_watch_thread_;
    std::mt19937 action_rng_;
    std::unordered_map<std::string, std::string> open_payloads_;
    std::unordered_map<std::string, std::string> open_responses_;
    MetricEventJournal metric_events_;
    SingleMapTaskController task_controller_;
    std::function<bool(const SingleMapTaskController&, std::string&)>
        task_controller_receipt_writer_;
    bool started_ = false;
    bool shutdown_started_ = false;
    bool shutdown_completed_ = false;
    bool shutdown_succeeded_ = true;
    bool client_initialized_ = false;
    bool task_stop_requested_ = false;

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
    int64_t latest_episode_step_ = 0;
    int64_t current_episode_max_steps_ = 0;
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
