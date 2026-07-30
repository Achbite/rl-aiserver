#pragma once

#include "ai/astar_solver.h"
#include "ai/maze_reward.h"
#include "ai/onnx_inferencer.h"
#include "config/config_loader.h"
#include "maze.grpc.pb.h"
#include "metrics/episode_metrics.h"
#include "model/model_manifest.h"
#include "model/model_distributor_client.h"
#include "sample/sample_sender.h"
#include "session/session_manager.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class MazeServiceImpl final : public maze::MazeService::Service {
public:
    explicit MazeServiceImpl(const AIServerConfig& config);
    ~MazeServiceImpl();

    bool Start();
    void BeginShutdown();
    bool IsReady() const;

    grpc::Status Init(grpc::ServerContext* ctx,
                      const maze::InitReq* req,
                      maze::InitRsp* rsp) override;

    grpc::Status BeginEpisode(grpc::ServerContext* ctx,
                              const maze::BeginEpisodeReq* req,
                              maze::EpisodeLifecycleRsp* rsp) override;

    grpc::Status Update(grpc::ServerContext* ctx,
                        const maze::UpdateReq* req,
                        maze::UpdateRsp* rsp) override;

    grpc::Status EndEpisode(grpc::ServerContext* ctx,
                            const maze::EpisodeEndReq* req,
                            maze::EpisodeEndRsp* rsp) override;

    grpc::Status AbortEpisode(grpc::ServerContext* ctx,
                              const maze::AbortEpisodeReq* req,
                              maze::EpisodeLifecycleRsp* rsp) override;

    grpc::Status GetAIServerStatus(grpc::ServerContext* ctx,
                                   const maze::AIServerStatusReq* req,
                                   maze::AIServerStatusRsp* rsp) override;

private:
    bool LoadInitialModel();
    void StartModelWatcher();
    void StopModelWatcher();
    void ModelWatchLoop();
    bool ActivateStagedModel();
    bool AtGlobalFragmentBoundary();
    bool CanActivateStagedModel();
    void InitAgentSolver(SessionManager::AgentRuntime& agent,
                         const SessionManager::Session& session);
    void ResetEpisodeState(SessionManager::Session& session, int episode_id);
    bool FinalizePendingTransition(SessionManager::Session& session,
                                   int agent_id,
                                   int gx,
                                   int gy,
                                   bool is_done,
                                   maze::TerminationReason reason);
    bool InferStateValue(const SessionManager::Session& session,
                         int gx,
                         int gy,
                         const std::vector<float>& client_obs,
                         float& value);
    bool ChooseModelAction(SessionManager::Session& session,
                           SessionManager::AgentRuntime& agent,
                           int gx,
                           int gy,
                           const std::vector<float>& client_obs,
                           int64_t action_frame_id,
                           int& action,
                           float& log_prob,
                           float& value);
    void BuildObs(const SessionManager::Session& session,
                  int gx,
                  int gy,
                  const std::vector<float>& client_obs,
                  std::vector<float>& obs) const;
    bool FlushAgentSamples(SessionManager::Session& session,
                           int agent_id,
                           bool is_episode_end,
                           maze::TerminationReason reason,
                           float bootstrap_value,
                           bool bootstrap_valid);
    void QuarantineAgentSamples(SessionManager::Session& session,
                                int agent_id);
    void FillSampleBatchMetadata(maze::SampleBatch& batch,
                                 const SessionManager::Session& session,
                                 int agent_id,
                                 maze::TerminationReason reason);
    int64_t CountCachedSamples();
    int64_t CountCachedFragments();
    int64_t EstimateCachedBytes();
    void RecordUpdateLatency(std::chrono::steady_clock::time_point start);
    void MarkDegraded(const std::string& error);
    static int64_t NowMs();
    static std::string CreateProducerInstanceId(const std::string& aiserver_id);

    AIServerConfig config_;
    SessionManager session_mgr_;
    mutable std::mutex mutex_;
    std::condition_variable model_condition_;
    SampleSender sample_sender_;
    EpisodeMetricsWindow episode_metrics_;
    OnnxInferencer onnx_inferencer_;
    ModelDistributorClient model_distributor_;
    ModelManifest model_manifest_;
    ModelManifest staged_model_manifest_;

    std::atomic<maze::AIServerState> state_{maze::AISERVER_STATE_STARTING};
    std::atomic<maze::ModelState> model_state_{maze::MODEL_STATE_WAITING};
    std::atomic<uint64_t> next_fragment_seq_{0};
    std::atomic<bool> model_watch_stop_{false};
    std::thread model_watch_thread_;
    std::string producer_instance_id_;
    bool started_ = false;
    bool shutdown_started_ = false;
    bool client_initialized_ = false;

    int64_t produced_unique_samples_ = 0;
    int64_t produced_unique_batches_ = 0;
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
    int64_t quarantined_sample_count_ = 0;
    int64_t quarantined_fragment_count_ = 0;
    std::string last_error_;
};
