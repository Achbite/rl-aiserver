#include "grpc/maze_service.h"

#include <filesystem>
#include <fstream>
#include <unistd.h>

MazeServiceImpl::MazeServiceImpl(const AIServerConfig& config)
    : runtime_(config, MazeTaskAdapter::RewardMetricPrefix()), task_adapter_(config) {
    task_adapter_.RegisterMetrics(runtime_.metric_registry_);
}
MazeServiceImpl::~MazeServiceImpl() = default;
bool MazeServiceImpl::Start() { return runtime_.Start(); }
bool MazeServiceImpl::BeginShutdown() { return runtime_.BeginShutdown(); }
bool MazeServiceImpl::IsReady() const { return runtime_.IsReady(); }
common::ServiceInstanceIdentity MazeServiceImpl::MetricSourceIdentity() const {
    return runtime_.MetricSourceIdentity();
}
training::GetMetricCatalogRsp MazeServiceImpl::MetricCatalog() const {
    return runtime_.MetricCatalog();
}
grpc::Status MazeServiceImpl::GetAIServerStatus(grpc::ServerContext* ctx,
    const training::AIServerStatusReq* req, training::AIServerStatusRsp* rsp) {
    return runtime_.GetAIServerStatus(ctx, req, rsp);
}

SingleMapModelIdentity MazeServiceImpl::ActiveModelIdentity() const {
    SingleMapModelIdentity identity;
    identity.model_step = runtime_.model_manifest_.model_step();
    identity.model_lineage_id = runtime_.model_manifest_.model_lineage_id();
    identity.trained_samples = runtime_.model_manifest_.trained_samples();
    return identity;
}

bool MazeServiceImpl::WriteTaskControllerReceipt(
    const SingleMapTaskController& controller,
    std::string& error) const {
    namespace fs = std::filesystem;
    const fs::path root(task_adapter_.Config().model.local_train_dir);
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

bool MazeServiceImpl::PrepareModelAction(
    SessionManager::Session& session, SessionManager::AgentRuntime& agent,
    int agent_id, int gx, int gy, int64_t frame, const std::vector<bool>& mask,
    uint64_t& segment_sequence, int64_t& activation_count, bool& latest_model_used,
    std::mt19937& rng, int& action, float& log_prob, float& value) {
    std::vector<float> observation;
    std::string error;
    if (!task_adapter_.BuildObservation(session, agent, gx, gy, frame, observation, error)) {
        runtime_.MarkDegraded("maze.observation construction failed: " + error);
        return false;
    }
    if (!runtime_.PrepareModelAction(session, agent, agent_id, std::move(observation),
            frame, mask, segment_sequence, activation_count, latest_model_used,
            rng, action, log_prob, value)) return false;
    agent.task.prev_grid_x = gx;
    agent.task.prev_grid_y = gy;
    return true;
}

bool MazeServiceImpl::FinalizePendingTransition(
    SessionManager::Session& session, int agent_id, int gx, int gy,
    bool is_done, maze::MazeTerminationReason reason,
    bool collect_training_sample, std::string& error) {
    auto agent_it = session.agents.find(agent_id);
    if (agent_it == session.agents.end()) {
        error = "transition Agent identity is unknown";
        return false;
    }
    auto& agent = agent_it->second;
    if (!agent.has_pending_action) return true;
    MazeRewardDetail reward;
    std::vector<float> next_observation;
    if (collect_training_sample) {
        reward = task_adapter_.CalculateReward(session, agent, gx, gy, is_done, reason);
        if (!reward.valid) { error = reward.error; return false; }
        if (!task_adapter_.BuildObservation(session, agent, gx, gy,
                agent.pending_action_frame_id + 1, next_observation, error)) {
            error = "next observation construction failed: " + error;
            return false;
        }
    }
    if (!runtime_.FinalizePendingTransition(session, agent_id, reward,
            next_observation, collect_training_sample, error)) return false;
    if (collect_training_sample) {
        for (const auto& item : reward.items) {
            if (item.first == "first_visit_bonus") agent.task.first_visit_bonus_total += item.second;
        }
    }
    if (is_done) agent.task.final_termination_reason = reason;
    return true;
}
