#pragma once

#include "runtime/training_runtime.h"
#include "task/maze_task_adapter.h"
#include "task/single_map_task_controller.h"
#include "proto/tasks/maze/task.grpc.pb.h"

class MazeServiceImpl final : public maze::MazeTaskService::Service,
                              public training::AIServerTrainingStatusService::Service {
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

    MetricEventService& Metrics() { return runtime_.metric_service_; }

    common::ServiceInstanceIdentity MetricSourceIdentity() const;
    training::GetMetricCatalogRsp MetricCatalog() const;

private:
    using SessionManager = MazeSessionManager;
    TrainingRuntime<MazeSessionManager> runtime_;
    MazeTaskAdapter task_adapter_;
    SingleMapTaskController task_controller_;
    SingleMapModelIdentity ActiveModelIdentity() const;
    bool WriteTaskControllerReceipt(const SingleMapTaskController&, std::string&) const;
    bool FinalizePendingTransition(SessionManager::Session&, int agent_id,
        int gx, int gy, bool is_done, maze::MazeTerminationReason reason,
        bool collect_training_sample, std::string& error);
    bool PrepareModelAction(SessionManager::Session&, SessionManager::AgentRuntime&,
        int agent_id, int gx, int gy, int64_t frame, const std::vector<bool>& mask,
        uint64_t& segment_sequence, int64_t& activation_count, bool& latest_model_used,
        std::mt19937& rng, int& action, float& log_prob, float& value);
};
