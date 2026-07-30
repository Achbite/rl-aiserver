#include "grpc/maze_service.h"

#include "log/logger.h"
#include "model/fragment_boundary.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
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

bool IsTerminalReason(maze::TerminationReason reason) {
    return reason == maze::TERMINATION_REASON_GOAL_REACHED ||
           reason == maze::TERMINATION_REASON_TIME_LIMIT ||
           reason == maze::TERMINATION_REASON_COUNTDOWN ||
           reason == maze::TERMINATION_REASON_CLIENT_ABORT ||
           reason == maze::TERMINATION_REASON_CHAIN_FAILURE;
}

}  // namespace

MazeServiceImpl::MazeServiceImpl(const AIServerConfig& config)
    : config_(config),
      sample_sender_(config.sample_output),
      model_distributor_(config.model_distribution, config.model),
      producer_instance_id_(
          CreateProducerInstanceId(config.sample_output.aiserver_id)) {}

MazeServiceImpl::~MazeServiceImpl() {
    BeginShutdown();
}

int64_t MazeServiceImpl::NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string MazeServiceImpl::CreateProducerInstanceId(
    const std::string& aiserver_id) {
    std::ostringstream output;
    output << aiserver_id << "-" << NowMs() << "-" << getpid();
    return output.str();
}

bool MazeServiceImpl::LoadInitialModel() {
    model_state_.store(maze::MODEL_STATE_WAITING);

    if (config_.server.run_mode == aiserver_mode::kModelEvaluation) {
        std::string path = config_.model.local_dir + "/" +
                           config_.model.save_name + ".onnx";
        std::string error;
        if (!onnx_inferencer_.LoadModel(
                path, config_.model.expected_obs_dim,
                config_.model.expected_action_dim, &error)) {
            model_state_.store(maze::MODEL_STATE_FAILED);
            last_error_ = "local model validation failed: " + error;
            return false;
        }
        model_manifest_.model_path = path;
        model_manifest_.model_version = 0;
        model_manifest_.run_id = config_.sample_output.run_id;
        if (!ComputeFileSha256(path, model_manifest_.sha256, error)) {
            model_state_.store(maze::MODEL_STATE_FAILED);
            last_error_ = "local model checksum failed: " + error;
            return false;
        }
        model_manifest_.ready = true;
        model_state_.store(maze::MODEL_STATE_READY);
        return true;
    }

    if (config_.server.run_mode == aiserver_mode::kLocalTest) {
        const std::string manifest_path =
            config_.model.smoke_dir + "/" + config_.model.manifest_name;
        std::string error;
        if (!LoadModelManifestFile(
                config_.model, manifest_path,
                "inference-smoke-fixture", model_manifest_, error)) {
            model_state_.store(maze::MODEL_STATE_FAILED);
            last_error_ = "smoke model manifest failed: " + error;
            return false;
        }
        if (!onnx_inferencer_.LoadModel(
                model_manifest_.model_path,
                config_.model.expected_obs_dim,
                config_.model.expected_action_dim, &error)) {
            model_state_.store(maze::MODEL_STATE_FAILED);
            last_error_ = "smoke model validation failed: " + error;
            return false;
        }
        model_state_.store(maze::MODEL_STATE_READY);
        return true;
    }

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(config_.model.startup_timeout_ms);
    std::string error;
    while (std::chrono::steady_clock::now() < deadline) {
        ModelManifest candidate;
        if (model_distributor_.FetchLatest(
                config_.sample_output.run_id,
                config_.sample_output.aiserver_id,
                candidate, error)) {
            std::string load_error;
            if (!onnx_inferencer_.LoadModel(
                    candidate.model_path,
                    config_.model.expected_obs_dim,
                    config_.model.expected_action_dim,
                    &load_error)) {
                std::string ack_error;
                model_distributor_.Ack(
                    candidate, config_.sample_output.run_id,
                    config_.sample_output.aiserver_id,
                    maze::MODEL_LOAD_STATUS_FAILED, load_error, ack_error);
                model_state_.store(maze::MODEL_STATE_FAILED);
                last_error_ = "ONNX model validation failed: " + load_error;
                return false;
            }
            std::string ack_error;
            if (!model_distributor_.Ack(
                    candidate, config_.sample_output.run_id,
                    config_.sample_output.aiserver_id,
                    maze::MODEL_LOAD_STATUS_LOADED, "loaded", ack_error)) {
                error = "model ACK failed: " + ack_error;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            model_manifest_ = std::move(candidate);
            model_state_.store(maze::MODEL_STATE_READY);
            LOG_INFO("MazeService",
                     "初始模型就绪: run=%s version=%d sha256=%s",
                     model_manifest_.run_id.c_str(),
                     model_manifest_.model_version,
                     model_manifest_.sha256.c_str());
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    model_state_.store(maze::MODEL_STATE_FAILED);
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
    model_condition_.notify_all();
    if (model_watch_thread_.joinable()) {
        model_watch_thread_.join();
    }
}

void MazeServiceImpl::ModelWatchLoop() {
    const auto poll_interval = std::chrono::milliseconds(
        std::max(50, config_.model_distribution.poll_interval_ms));
    while (!model_watch_stop_.load()) {
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
                config_.sample_output.run_id,
                config_.sample_output.aiserver_id,
                latest_version, latest_checksum, error) &&
            latest_version > active_version &&
            latest_version > staged_version) {
            const int requested_version = active_version + 1;
            ModelManifest candidate;
            if (model_distributor_.FetchVersion(
                    config_.sample_output.run_id,
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
                        candidate, config_.sample_output.run_id,
                        config_.sample_output.aiserver_id,
                        maze::MODEL_LOAD_STATUS_FAILED,
                        validation_error, ack_error);
                    std::lock_guard<std::mutex> lock(mutex_);
                    last_error_ =
                        "staged model validation failed: " +
                        validation_error;
                } else {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (candidate.model_version ==
                            model_manifest_.model_version + 1 &&
                        candidate.model_version >
                            staged_model_manifest_.model_version) {
                        staged_model_manifest_ = std::move(candidate);
                        LOG_INFO(
                            "MazeService",
                            "模型已暂存: version=%d sha256=%s",
                            staged_model_manifest_.model_version,
                            staged_model_manifest_.sha256.c_str());
                        if (CanActivateStagedModel()) {
                            ActivateStagedModel();
                        }
                        model_condition_.notify_all();
                    }
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

bool MazeServiceImpl::Start() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (started_) return state_.load() == maze::AISERVER_STATE_READY;

        state_.store(maze::AISERVER_STATE_STARTING);
        if (config_.server.run_mode == aiserver_mode::kTraining) {
            if (!sample_sender_.Start()) {
                auto sender = sample_sender_.GetSnapshot();
                last_error_ = sender.last_error;
                state_.store(maze::AISERVER_STATE_DEGRADED);
                return false;
            }
            if (!LoadInitialModel()) {
                state_.store(maze::AISERVER_STATE_DEGRADED);
                return false;
            }
        } else if (config_.server.run_mode ==
                       aiserver_mode::kModelEvaluation ||
                   config_.server.run_mode == aiserver_mode::kLocalTest) {
            if (!LoadInitialModel()) {
                state_.store(maze::AISERVER_STATE_DEGRADED);
                return false;
            }
        }

        started_ = true;
        state_.store(maze::AISERVER_STATE_READY);
    }
    StartModelWatcher();
    return true;
}

bool MazeServiceImpl::IsReady() const {
    return state_.load() == maze::AISERVER_STATE_READY &&
           (config_.server.run_mode == aiserver_mode::kAstarTest ||
            model_state_.load() == maze::MODEL_STATE_READY) &&
           (config_.server.run_mode != aiserver_mode::kTraining ||
            (sample_sender_.IsReady() && !sample_sender_.IsDegraded()));
}

void MazeServiceImpl::BeginShutdown() {
    StopModelWatcher();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_started_) return;
        shutdown_started_ = true;
        state_.store(maze::AISERVER_STATE_DRAINING);

        SessionManager::Session* session = session_mgr_.GetSession(0);
        if (session) {
            for (auto& item : session->agents) {
                int agent_id = item.first;
                QuarantineAgentSamples(*session, agent_id);
            }
        }
    }

    if (config_.server.run_mode == aiserver_mode::kTraining) {
        sample_sender_.StopAndDrain();
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        SessionManager::Session* session = session_mgr_.GetSession(0);
        if (session) {
            int64_t remaining_samples = CountCachedSamples();
            int64_t remaining_batches = CountCachedFragments();
            if (remaining_samples > 0) {
                sample_sender_.RecordFinalDrop(
                    remaining_samples, remaining_batches,
                    "session cache remained after drain deadline");
            }
            session->agent_sample_caches.clear();
            session->pending_sample_batches.clear();
        }
        state_.store(maze::AISERVER_STATE_STOPPED);
    }
}

bool MazeServiceImpl::ValidateRunId(const std::string& run_id,
                                    std::string& error) const {
    if (run_id.empty()) {
        error = "run_id is required";
        return false;
    }
    if (run_id != config_.sample_output.run_id) {
        error = "run_id does not match AIServer";
        return false;
    }
    return true;
}

bool MazeServiceImpl::AtGlobalFragmentBoundary() {
    SessionManager::Session* session = session_mgr_.GetSession(0);
    if (!session ||
        session->episode_state != SessionManager::EpisodeState::Active) {
        return true;
    }
    std::vector<FragmentBoundaryState> agents;
    agents.reserve(session->agents.size());
    for (const auto& item : session->agents) {
        const int agent_id = item.first;
        const auto& agent = item.second;
        const auto cache = session->agent_sample_caches.find(agent_id);
        agents.push_back(
            FragmentBoundaryState{
                agent.has_pending_action,
                cache != session->agent_sample_caches.end() &&
                    !cache->second.empty(),
                session->pending_sample_batches.find(agent_id) !=
                    session->pending_sample_batches.end(),
            });
    }
    return AllAgentsAtFragmentBoundary(agents);
}

bool MazeServiceImpl::CanActivateStagedModel() {
    return staged_model_manifest_.model_version >= 0 &&
           AtGlobalFragmentBoundary();
}

bool MazeServiceImpl::ActivateStagedModel() {
    if (!CanActivateStagedModel()) return true;

    const ModelManifest candidate = staged_model_manifest_;
    if (candidate.model_version != model_manifest_.model_version + 1) {
        MarkDegraded("staged model version is not contiguous");
        return false;
    }

    std::string error;
    if (!onnx_inferencer_.LoadModel(
            candidate.model_path,
            config_.model.expected_obs_dim,
            config_.model.expected_action_dim, &error)) {
        std::string ack_error;
        model_distributor_.Ack(
            candidate, config_.sample_output.run_id,
            config_.sample_output.aiserver_id,
            maze::MODEL_LOAD_STATUS_FAILED, error, ack_error);
        MarkDegraded("staged model activation failed: " + error);
        return false;
    }

    std::string ack_error;
    if (!model_distributor_.Ack(
            candidate, config_.sample_output.run_id,
            config_.sample_output.aiserver_id,
            maze::MODEL_LOAD_STATUS_LOADED, "loaded", ack_error)) {
        std::string rollback_error;
        onnx_inferencer_.LoadModel(
            model_manifest_.model_path,
            config_.model.expected_obs_dim,
            config_.model.expected_action_dim, &rollback_error);
        MarkDegraded("model ACK failed: " + ack_error);
        return false;
    }

    model_manifest_ = candidate;
    staged_model_manifest_ = ModelManifest{};
    ++model_switch_count_;
    LOG_INFO(
        "MazeService", "模型切换完成: version=%d sha256=%s",
        model_manifest_.model_version, model_manifest_.sha256.c_str());
    return true;
}

void MazeServiceImpl::InitAgentSolver(
    SessionManager::AgentRuntime& agent,
    const SessionManager::Session& session) {
    agent.solver.Init(session.map_width, session.map_height,
                      config_.strategy.grid_size);
    agent.solver.AddWall(5000, 0, 5000, 14000, 100);
    agent.solver.AddWall(10000, 6000, 10000, 20000, 100);
    agent.solver.AddWall(15000, 0, 15000, 14000, 100);
    agent.path_valid = agent.solver.PlanPath(
        session.start_x, session.start_y, session.end_x, session.end_y);
}

void MazeServiceImpl::ResetEpisodeState(SessionManager::Session& session,
                                        int episode_id) {
    session.current_episode_id = episode_id;
    session.episode_state = SessionManager::EpisodeState::Active;
    session.episode_history[episode_id] = SessionManager::EpisodeState::Active;
    session.last_frame_id = -1;
    session.last_actions.clear();
    session.first_done_frame = -1;
    session.ranking_order.clear();
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
        agent.pending_obs.clear();
        agent.fragment_model_version = -1;
        agent.fragment_model_checksum.clear();
        agent.fragment_first_action_frame_id = -1;
        agent.visited.clear();
        agent.recent_positions.clear();
        if (config_.server.run_mode == aiserver_mode::kAstarTest) {
            agent.path_valid = agent.solver.PlanPath(
                session.start_x, session.start_y,
                session.end_x, session.end_y);
        }
    }
}

void MazeServiceImpl::BuildObs(
    const SessionManager::Session& session,
    int gx,
    int gy,
    const std::vector<float>& client_obs,
    std::vector<float>& obs) const {
    obs.assign(static_cast<std::size_t>(config_.model.expected_obs_dim), 0.0f);
    if (obs.size() < 5) return;

    obs[0] = session.grid_cols > 0
                 ? static_cast<float>(gx) / session.grid_cols
                 : 0.0f;
    obs[1] = session.grid_rows > 0
                 ? static_cast<float>(gy) / session.grid_rows
                 : 0.0f;
    float dx = static_cast<float>(session.end_gx - gx);
    float dy = static_cast<float>(session.end_gy - gy);
    float distance = std::sqrt(dx * dx + dy * dy);
    if (distance > 0.0f) {
        obs[2] = dx / distance;
        obs[3] = dy / distance;
    }
    float max_distance = std::sqrt(static_cast<float>(
        session.grid_cols * session.grid_cols +
        session.grid_rows * session.grid_rows));
    obs[4] = max_distance > 0.0f ? distance / max_distance : 0.0f;

    std::size_t ray_count =
        std::min(client_obs.size(), obs.size() > 5 ? obs.size() - 5 : 0);
    for (std::size_t i = 0; i < ray_count; ++i) {
        obs[5 + i] = client_obs[i];
    }
}

bool MazeServiceImpl::ChooseModelAction(
    SessionManager::Session& session,
    SessionManager::AgentRuntime& agent,
    int gx,
    int gy,
    const std::vector<float>& client_obs,
    int64_t action_frame_id,
    int& action,
    float& log_prob,
    float& value) {
    std::vector<float> obs;
    BuildObs(session, gx, gy, client_obs, obs);
    std::vector<float> probabilities;

    auto start = std::chrono::steady_clock::now();
    bool inferred = onnx_inferencer_.Infer(
        obs, static_cast<int>(obs.size()), probabilities, value);
    double latency_ms = ElapsedMs(start);
    ++inference_count_;
    inference_latency_sum_ms_ += latency_ms;
    inference_latency_max_ms_ =
        std::max(inference_latency_max_ms_, latency_ms);

    if (!inferred ||
        probabilities.size() !=
            static_cast<std::size_t>(config_.model.expected_action_dim)) {
        MarkDegraded("ONNX inference failed");
        return false;
    }

    double sum = 0.0;
    for (float probability : probabilities) {
        if (!std::isfinite(probability) || probability < 0.0f) {
            MarkDegraded("ONNX action probabilities are invalid");
            return false;
        }
        sum += probability;
    }
    if (!(sum > 0.0)) {
        MarkDegraded("ONNX action probability sum is zero");
        return false;
    }

    static thread_local std::mt19937 generator(std::random_device{}());
    std::discrete_distribution<int> distribution(
        probabilities.begin(), probabilities.end());
    action = distribution(generator);
    log_prob = std::log(
        std::max(probabilities[static_cast<std::size_t>(action)], 1e-8f));

    if (config_.server.run_mode == aiserver_mode::kTraining) {
        agent.pending_obs = std::move(obs);
        agent.pending_action = action;
        agent.pending_action_frame_id = action_frame_id;
        agent.pending_log_prob = log_prob;
        agent.pending_value = value;
        agent.pending_model_version = model_manifest_.model_version;
        agent.pending_model_checksum = model_manifest_.sha256;
        agent.has_pending_action = true;
    }
    agent.prev_grid_x = gx;
    agent.prev_grid_y = gy;
    return true;
}

bool MazeServiceImpl::InferStateValue(
    const SessionManager::Session& session,
    int gx,
    int gy,
    const std::vector<float>& client_obs,
    float& value) {
    std::vector<float> obs;
    BuildObs(session, gx, gy, client_obs, obs);
    std::vector<float> probabilities;
    auto start = std::chrono::steady_clock::now();
    const bool inferred = onnx_inferencer_.Infer(
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
    maze::TerminationReason reason) {
    auto agent_it = session.agents.find(agent_id);
    if (agent_it == session.agents.end()) return false;
    auto& agent = agent_it->second;
    if (!agent.has_pending_action) return true;
    auto& cache = session.agent_sample_caches[agent_id];
    if (cache.empty()) {
        agent.fragment_model_version = agent.pending_model_version;
        agent.fragment_model_checksum = agent.pending_model_checksum;
        agent.fragment_first_action_frame_id =
            agent.pending_action_frame_id;
    } else if (agent.fragment_model_version !=
                   agent.pending_model_version ||
               agent.fragment_model_checksum !=
                   agent.pending_model_checksum) {
        MarkDegraded("fragment contains mixed behavior models");
        return false;
    }

    RewardDetail reward = MazeReward::Calculate(
        session, agent_id, gx, gy, is_done,
        static_cast<int>(session.agents.size()));

    maze::Sample sample;
    for (float value : agent.pending_obs) sample.add_obs(value);
    sample.set_action(agent.pending_action);
    sample.set_reward(reward.total);
    sample.set_old_log_prob(agent.pending_log_prob);
    sample.set_old_vpred(agent.pending_value);
    bool terminated =
        is_done && reason == maze::TERMINATION_REASON_GOAL_REACHED;
    sample.set_terminated(terminated);
    sample.set_truncated(is_done && !terminated);
    sample.set_termination_reason(
        is_done ? reason : maze::TERMINATION_REASON_ACTIVE);
    sample.set_action_frame_id(agent.pending_action_frame_id);
    for (const auto& item : reward.items) {
        (*sample.mutable_reward_details())[item.first] = item.second;
    }

    cache.push_back(std::move(sample));
    ++produced_unique_samples_;
    agent.has_pending_action = false;
    agent.pending_action_frame_id = -1;
    agent.pending_model_version = -1;
    agent.pending_model_checksum.clear();
    agent.pending_obs.clear();

    int key = gy * session.grid_cols + gx;
    agent.visited.insert(key);
    agent.recent_positions.push_back(key);
    if (agent.recent_positions.size() > 8) {
        agent.recent_positions.pop_front();
    }

    return true;
}

void MazeServiceImpl::FillSampleBatchMetadata(
    maze::SampleBatch& batch,
    const SessionManager::Session& session,
    int agent_id,
    maze::TerminationReason reason) {
    uint64_t sequence = next_fragment_seq_.fetch_add(1);
    const auto agent_it = session.agents.find(agent_id);
    if (agent_it == session.agents.end()) return;
    const auto& agent = agent_it->second;
    batch.set_protocol_version(2);
    batch.set_run_id(config_.sample_output.run_id);
    batch.set_aiserver_id(config_.sample_output.aiserver_id);
    batch.set_env_id(session.env_id);
    batch.set_session_id(session.session_id);
    batch.set_episode_id(session.current_episode_id);
    batch.set_agent_id(agent_id);
    batch.set_fragment_seq(sequence);
    batch.set_fragment_id(
        static_cast<int32_t>(sequence %
                             static_cast<uint64_t>(
                                 std::numeric_limits<int32_t>::max())));
    batch.set_producer_instance_id(producer_instance_id_);
    batch.set_behavior_model_version(agent.fragment_model_version);
    batch.set_behavior_model_checksum(agent.fragment_model_checksum);
    batch.set_created_ts_ms(NowMs());
    batch.set_termination_reason(reason);

    std::ostringstream batch_id;
    batch_id << config_.sample_output.run_id << "/"
             << producer_instance_id_ << "/"
             << session.session_id << "/"
             << session.current_episode_id << "/"
             << agent_id << "/" << sequence;
    batch.set_batch_id(batch_id.str());
}

bool MazeServiceImpl::FlushAgentSamples(
    SessionManager::Session& session,
    int agent_id,
    bool is_episode_end,
    maze::TerminationReason reason,
    float bootstrap_value,
    bool bootstrap_valid) {
    auto& cache = session.agent_sample_caches[agent_id];
    if (cache.empty()) return true;
    auto agent_it = session.agents.find(agent_id);
    if (agent_it == session.agents.end()) return false;
    auto& agent = agent_it->second;

    auto pending_it = session.pending_sample_batches.find(agent_id);
    if (pending_it == session.pending_sample_batches.end()) {
        maze::SampleBatch batch;
        batch.set_is_episode_end(is_episode_end);
        FillSampleBatchMetadata(batch, session, agent_id, reason);
        batch.set_bootstrap_value(bootstrap_value);
        batch.set_bootstrap_valid(bootstrap_valid);
        batch.set_first_action_frame_id(
            agent.fragment_first_action_frame_id);
        batch.set_last_action_frame_id(
            cache.back().action_frame_id());
        for (const auto& sample : cache) {
            *batch.add_samples() = sample;
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
    agent.fragment_first_action_frame_id = -1;
    return true;
}

void MazeServiceImpl::QuarantineAgentSamples(
    SessionManager::Session& session,
    int agent_id) {
    auto agent_it = session.agents.find(agent_id);
    if (agent_it == session.agents.end()) return;
    auto& agent = agent_it->second;
    auto& cache = session.agent_sample_caches[agent_id];
    const int64_t discarded =
        static_cast<int64_t>(cache.size()) +
        (agent.has_pending_action ? 1 : 0);
    if (discarded > 0) {
        quarantined_sample_count_ += discarded;
        ++quarantined_fragment_count_;
    }
    produced_unique_samples_ -= static_cast<int64_t>(cache.size());
    cache.clear();
    session.pending_sample_batches.erase(agent_id);
    agent.has_pending_action = false;
    agent.pending_action_frame_id = -1;
    agent.pending_model_version = -1;
    agent.pending_model_checksum.clear();
    agent.pending_obs.clear();
    agent.fragment_model_version = -1;
    agent.fragment_model_checksum.clear();
    agent.fragment_first_action_frame_id = -1;
}

grpc::Status MazeServiceImpl::Init(
    grpc::ServerContext*,
    const maze::InitReq* req,
    maze::InitRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string error;
    if (!IsReady()) {
        rsp->set_ret_code(-1);
        rsp->set_result(maze::LIFECYCLE_RESULT_REJECTED);
        rsp->set_message("AIServer is not ready");
        return grpc::Status::OK;
    }
    if (!ValidateRunId(req->run_id(), error)) {
        rsp->set_ret_code(-1);
        rsp->set_result(maze::LIFECYCLE_RESULT_REJECTED);
        rsp->set_message(error);
        return grpc::Status::OK;
    }
    if (req->agent_num() <= 0 ||
        req->agent_num() > config_.server.max_agents) {
        rsp->set_ret_code(-1);
        rsp->set_result(maze::LIFECYCLE_RESULT_REJECTED);
        rsp->set_message("agent_num is outside the configured limit");
        return grpc::Status::OK;
    }

    auto* session = session_mgr_.GetOrCreateSession(req->session_id());
    if (session->initialized) {
        bool same_identity =
            session->run_id == req->run_id() &&
            session->client_id == req->client_id() &&
            session->env_id == req->env_id() &&
            session->agents.size() ==
                static_cast<std::size_t>(req->agent_num());
        rsp->set_ret_code(same_identity ? 0 : -1);
        rsp->set_result(
            same_identity ? maze::LIFECYCLE_RESULT_ALREADY_APPLIED
                          : maze::LIFECYCLE_RESULT_REJECTED);
        rsp->set_message(
            same_identity ? "session already initialized"
                          : "session identity does not match existing session");
        return grpc::Status::OK;
    }

    session->run_id = req->run_id();
    session->client_id = req->client_id();
    session->env_id = req->env_id();
    session->map_width = req->map_size().x();
    session->map_height = req->map_size().y();
    session->start_x = req->start_pos().x();
    session->start_y = req->start_pos().y();
    session->end_x = req->end_pos().x();
    session->end_y = req->end_pos().y();
    session->grid_size =
        req->grid_size() > 0.0f
            ? req->grid_size()
            : static_cast<float>(config_.strategy.grid_size);
    session->grid_cols =
        req->grid_cols() > 0
            ? req->grid_cols()
            : static_cast<int>(
                  std::ceil(session->map_width / session->grid_size));
    session->grid_rows =
        req->grid_rows() > 0
            ? req->grid_rows()
            : static_cast<int>(
                  std::ceil(session->map_height / session->grid_size));
    session->end_gx =
        req->has_end_grid()
            ? static_cast<int>(req->end_grid().x())
            : static_cast<int>(session->end_x / session->grid_size);
    session->end_gy =
        req->has_end_grid()
            ? static_cast<int>(req->end_grid().y())
            : static_cast<int>(session->end_y / session->grid_size);
    int grid_size = static_cast<int>(std::round(session->grid_size));

    for (int agent_id = 0; agent_id < req->agent_num(); ++agent_id) {
        auto& agent = session->agents[agent_id];
        if (config_.server.run_mode == aiserver_mode::kAstarTest) {
            InitAgentSolver(agent, *session);
        }
    }
    if (config_.server.run_mode == aiserver_mode::kAstarTest) {
        session->InitBlocked(grid_size);
    }
    session->initialized = true;
    client_initialized_ = true;

    rsp->set_ret_code(0);
    rsp->set_result(maze::LIFECYCLE_RESULT_OK);
    rsp->set_message("session initialized");
    LOG_INFO("MazeService",
             "Client 初始化完成: run=%s session=%d client=%s env=%s agents=%d",
             session->run_id.c_str(), session->session_id,
             session->client_id.c_str(), session->env_id.c_str(),
             req->agent_num());
    return grpc::Status::OK;
}

grpc::Status MazeServiceImpl::BeginEpisode(
    grpc::ServerContext*,
    const maze::BeginEpisodeReq* req,
    maze::EpisodeLifecycleRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string error;
    if (!ValidateRunId(req->run_id(), error)) {
        rsp->set_ret_code(-1);
        rsp->set_result(maze::LIFECYCLE_RESULT_REJECTED);
        rsp->set_message(error);
        return grpc::Status::OK;
    }
    auto* session = session_mgr_.GetSession(req->session_id());
    if (!session || !session->initialized) {
        rsp->set_ret_code(-1);
        rsp->set_result(maze::LIFECYCLE_RESULT_REJECTED);
        rsp->set_message("session is not initialized");
        return grpc::Status::OK;
    }

    auto history = session->episode_history.find(req->episode_id());
    if (history != session->episode_history.end()) {
        rsp->set_ret_code(0);
        rsp->set_result(maze::LIFECYCLE_RESULT_ALREADY_APPLIED);
        rsp->set_message("episode identity was already applied");
        rsp->set_episode_id(req->episode_id());
        return grpc::Status::OK;
    }
    if (session->episode_state == SessionManager::EpisodeState::Active) {
        rsp->set_ret_code(-1);
        rsp->set_result(maze::LIFECYCLE_RESULT_REJECTED);
        rsp->set_message("another episode is active");
        return grpc::Status::OK;
    }

    if (config_.server.run_mode == aiserver_mode::kTraining &&
        !ActivateStagedModel()) {
        rsp->set_ret_code(-1);
        rsp->set_result(maze::LIFECYCLE_RESULT_REJECTED);
        rsp->set_message("staged model could not be activated");
        return grpc::Status::OK;
    }
    ResetEpisodeState(*session, req->episode_id());
    rsp->set_ret_code(0);
    rsp->set_result(maze::LIFECYCLE_RESULT_OK);
    rsp->set_message("episode started");
    rsp->set_episode_id(req->episode_id());
    return grpc::Status::OK;
}

void MazeServiceImpl::RecordUpdateLatency(
    std::chrono::steady_clock::time_point start) {
    double latency_ms = ElapsedMs(start);
    ++update_rpc_count_;
    update_rpc_latency_sum_ms_ += latency_ms;
    update_rpc_latency_max_ms_ =
        std::max(update_rpc_latency_max_ms_, latency_ms);
}

void MazeServiceImpl::MarkDegraded(const std::string& error) {
    state_.store(maze::AISERVER_STATE_DEGRADED);
    last_error_ = error;
    sample_sender_.MarkDegraded(error);
}

grpc::Status MazeServiceImpl::Update(
    grpc::ServerContext*,
    const maze::UpdateReq* req,
    maze::UpdateRsp* rsp) {
    auto rpc_start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);

    if (sample_sender_.IsDegraded()) {
        auto sender = sample_sender_.GetSnapshot();
        MarkDegraded(sender.last_error);
    }
    if (!IsReady()) {
        RecordUpdateLatency(rpc_start);
        return grpc::Status(
            grpc::StatusCode::UNAVAILABLE, "AIServer sample chain is degraded");
    }

    std::string error;
    if (!ValidateRunId(req->run_id(), error)) {
        RecordUpdateLatency(rpc_start);
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, error);
    }
    auto* session = session_mgr_.GetSession(req->session_id());
    if (!session || !session->initialized) {
        RecordUpdateLatency(rpc_start);
        return grpc::Status(
            grpc::StatusCode::FAILED_PRECONDITION,
            "session is not initialized");
    }
    if (session->episode_state != SessionManager::EpisodeState::Active ||
        session->current_episode_id != req->episode_id()) {
        RecordUpdateLatency(rpc_start);
        return grpc::Status(
            grpc::StatusCode::FAILED_PRECONDITION,
            "episode is not active");
    }

    if (req->frame_id() == session->last_frame_id) {
        for (const auto& cached : session->last_actions) {
            *rsp->add_actions() = cached;
        }
        rsp->set_replayed(true);
        RecordUpdateLatency(rpc_start);
        return grpc::Status::OK;
    }
    if (req->frame_id() != session->last_frame_id + 1) {
        RecordUpdateLatency(rpc_start);
        return grpc::Status(
            grpc::StatusCode::OUT_OF_RANGE,
            req->frame_id() < session->last_frame_id
                ? "frame_id is stale"
                : "frame_id is not contiguous");
    }
    if (req->agents_size() !=
        static_cast<int>(session->agents.size())) {
        RecordUpdateLatency(rpc_start);
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            "Update must contain every initialized agent");
    }

    std::unordered_set<int> seen_agents;
    for (const auto& state : req->agents()) {
        auto agent_it = session->agents.find(state.agent_id());
        if (agent_it == session->agents.end()) {
            RecordUpdateLatency(rpc_start);
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                "agent_id was not initialized");
        }
        if (!seen_agents.insert(state.agent_id()).second) {
            RecordUpdateLatency(rpc_start);
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                "Update contains a duplicate agent_id");
        }
        if (state.is_done() && !IsTerminalReason(state.termination_reason())) {
            RecordUpdateLatency(rpc_start);
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                "done agent requires a terminal reason");
        }
        if (!state.is_done() &&
            state.termination_reason() !=
                maze::TERMINATION_REASON_ACTIVE &&
            state.termination_reason() !=
                maze::TERMINATION_REASON_UNSPECIFIED) {
            RecordUpdateLatency(rpc_start);
            return grpc::Status(
                grpc::StatusCode::INVALID_ARGUMENT,
                "active agent has a terminal reason");
        }

        auto& agent = agent_it->second;
        if (state.is_done() && !agent.done_collected) {
            agent.done_collected = true;
            agent.reached_goal =
                state.termination_reason() ==
                maze::TERMINATION_REASON_GOAL_REACHED;
            session->ranking_order.push_back(state.agent_id());
            if (session->first_done_frame < 0) {
                session->first_done_frame = req->frame_id();
            }
        }
    }

    if (config_.server.run_mode == aiserver_mode::kTraining) {
        bool flushed_fragment = false;
        for (const auto& state : req->agents()) {
            const maze::TerminationReason reason =
                state.is_done()
                    ? state.termination_reason()
                    : maze::TERMINATION_REASON_ACTIVE;
            if (reason == maze::TERMINATION_REASON_CLIENT_ABORT ||
                reason == maze::TERMINATION_REASON_CHAIN_FAILURE) {
                QuarantineAgentSamples(*session, state.agent_id());
                continue;
            }
            if (!FinalizePendingTransition(
                    *session, state.agent_id(),
                    static_cast<int>(state.pos().x()),
                    static_cast<int>(state.pos().y()),
                    state.is_done(), reason)) {
                RecordUpdateLatency(rpc_start);
                return grpc::Status(
                    grpc::StatusCode::RESOURCE_EXHAUSTED,
                    "sample transition could not be finalized");
            }
        }

        for (const auto& state : req->agents()) {
            const maze::TerminationReason reason =
                state.is_done()
                    ? state.termination_reason()
                    : maze::TERMINATION_REASON_ACTIVE;
            if (reason == maze::TERMINATION_REASON_CLIENT_ABORT ||
                reason == maze::TERMINATION_REASON_CHAIN_FAILURE) {
                continue;
            }
            const auto cache =
                session->agent_sample_caches.find(state.agent_id());
            if (cache == session->agent_sample_caches.end() ||
                cache->second.empty()) {
                continue;
            }
            const bool fragment_full =
                cache->second.size() >= static_cast<std::size_t>(
                    config_.sample_output.fragment_samples);
            if (!state.is_done() && !fragment_full) continue;

            float bootstrap_value = 0.0f;
            bool bootstrap_valid = false;
            if (reason == maze::TERMINATION_REASON_GOAL_REACHED) {
                bootstrap_valid = true;
            } else {
                std::vector<float> client_obs(
                    state.obs().begin(), state.obs().end());
                if (!InferStateValue(
                        *session,
                        static_cast<int>(state.pos().x()),
                        static_cast<int>(state.pos().y()),
                        client_obs, bootstrap_value)) {
                    RecordUpdateLatency(rpc_start);
                    return grpc::Status(
                        grpc::StatusCode::INTERNAL,
                        "bootstrap value inference failed");
                }
                bootstrap_valid = true;
            }
            if (!FlushAgentSamples(
                    *session, state.agent_id(), state.is_done(),
                    reason, bootstrap_value, bootstrap_valid)) {
                RecordUpdateLatency(rpc_start);
                return grpc::Status(
                    grpc::StatusCode::RESOURCE_EXHAUSTED,
                    "sample outbound queue is not accepting fragments");
            }
            flushed_fragment = true;
        }

        if (flushed_fragment &&
            AtGlobalFragmentBoundary() &&
            staged_model_manifest_.model_version < 0 &&
            config_.model_distribution.boundary_wait_ms > 0) {
            const int active_version = model_manifest_.model_version;
            model_condition_.wait_for(
                lock,
                std::chrono::milliseconds(
                    config_.model_distribution.boundary_wait_ms),
                [this, active_version]() {
                    return model_watch_stop_.load() ||
                           staged_model_manifest_.model_version >= 0 ||
                           model_manifest_.model_version > active_version;
                });
        }
        if (!ActivateStagedModel()) {
            RecordUpdateLatency(rpc_start);
            return grpc::Status(
                grpc::StatusCode::UNAVAILABLE,
                "staged model could not be activated");
        }
    }

    std::vector<maze::AgentAction> actions;
    actions.reserve(static_cast<std::size_t>(req->agents_size()));
    for (const auto& state : req->agents()) {
        auto agent_it = session->agents.find(state.agent_id());
        auto& agent = agent_it->second;
        const int gx = static_cast<int>(state.pos().x());
        const int gy = static_cast<int>(state.pos().y());
        maze::AgentAction action;
        action.set_agent_id(state.agent_id());
        if (state.is_done()) {
            action.set_action_id(0);
            actions.push_back(action);
            continue;
        }

        int chosen_action = 0;
        if (config_.server.run_mode == aiserver_mode::kAstarTest) {
            if (agent.path_valid) {
                chosen_action = agent.solver.GetAction(gx, gy);
            }
        } else {
            std::vector<float> client_obs(
                state.obs().begin(), state.obs().end());
            float log_prob = 0.0f;
            float value = 0.0f;
            if (!ChooseModelAction(
                    *session, agent, gx, gy, client_obs,
                    req->frame_id(),
                    chosen_action, log_prob, value)) {
                RecordUpdateLatency(rpc_start);
                return grpc::Status(
                    grpc::StatusCode::INTERNAL,
                    "model inference failed");
            }
        }
        action.set_action_id(chosen_action);
        agent.last_action = chosen_action;
        actions.push_back(action);
    }

    session->last_frame_id = req->frame_id();
    session->last_actions = actions;
    for (const auto& action : actions) {
        *rsp->add_actions() = action;
    }
    rsp->set_replayed(false);
    RecordUpdateLatency(rpc_start);
    return grpc::Status::OK;
}

grpc::Status MazeServiceImpl::EndEpisode(
    grpc::ServerContext*,
    const maze::EpisodeEndReq* req,
    maze::EpisodeEndRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string error;
    if (!ValidateRunId(req->run_id(), error)) {
        rsp->set_ret_code(-1);
        rsp->set_result(maze::LIFECYCLE_RESULT_REJECTED);
        rsp->set_message(error);
        return grpc::Status::OK;
    }
    auto* session = session_mgr_.GetSession(req->session_id());
    if (!session) {
        rsp->set_ret_code(-1);
        rsp->set_result(maze::LIFECYCLE_RESULT_REJECTED);
        rsp->set_message("session does not exist");
        return grpc::Status::OK;
    }
    auto history = session->episode_history.find(req->episode_id());
    if (history != session->episode_history.end() &&
        history->second == SessionManager::EpisodeState::Ended) {
        rsp->set_ret_code(0);
        rsp->set_result(maze::LIFECYCLE_RESULT_ALREADY_APPLIED);
        rsp->set_message("episode already ended");
        return grpc::Status::OK;
    }
    if (session->episode_state != SessionManager::EpisodeState::Active ||
        session->current_episode_id != req->episode_id()) {
        rsp->set_ret_code(-1);
        rsp->set_result(maze::LIFECYCLE_RESULT_REJECTED);
        rsp->set_message("episode is not active");
        return grpc::Status::OK;
    }
    for (const auto& item : session->agents) {
        if (!item.second.done_collected ||
            (config_.server.run_mode == aiserver_mode::kTraining &&
             (item.second.has_pending_action ||
              !session->agent_sample_caches[item.first].empty() ||
              session->pending_sample_batches.find(item.first) !=
                  session->pending_sample_batches.end()))) {
            rsp->set_ret_code(-1);
            rsp->set_result(maze::LIFECYCLE_RESULT_REJECTED);
            rsp->set_message("terminal state has not been reported for every agent");
            return grpc::Status::OK;
        }
    }

    session->episode_state = SessionManager::EpisodeState::Ended;
    session->episode_history[req->episode_id()] =
        SessionManager::EpisodeState::Ended;
    rsp->set_ret_code(0);
    rsp->set_result(maze::LIFECYCLE_RESULT_OK);
    rsp->set_message("episode ended");
    return grpc::Status::OK;
}

grpc::Status MazeServiceImpl::AbortEpisode(
    grpc::ServerContext*,
    const maze::AbortEpisodeReq* req,
    maze::EpisodeLifecycleRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string error;
    if (!ValidateRunId(req->run_id(), error)) {
        rsp->set_ret_code(-1);
        rsp->set_result(maze::LIFECYCLE_RESULT_REJECTED);
        rsp->set_message(error);
        return grpc::Status::OK;
    }
    auto* session = session_mgr_.GetSession(req->session_id());
    if (!session) {
        rsp->set_ret_code(-1);
        rsp->set_result(maze::LIFECYCLE_RESULT_REJECTED);
        rsp->set_message("session does not exist");
        return grpc::Status::OK;
    }
    auto history = session->episode_history.find(req->episode_id());
    if (history != session->episode_history.end() &&
        history->second == SessionManager::EpisodeState::Aborted) {
        rsp->set_ret_code(0);
        rsp->set_result(maze::LIFECYCLE_RESULT_ALREADY_APPLIED);
        rsp->set_message("episode already aborted");
        rsp->set_episode_id(req->episode_id());
        return grpc::Status::OK;
    }
    if (session->episode_state != SessionManager::EpisodeState::Active ||
        session->current_episode_id != req->episode_id()) {
        rsp->set_ret_code(-1);
        rsp->set_result(maze::LIFECYCLE_RESULT_REJECTED);
        rsp->set_message("episode is not active");
        return grpc::Status::OK;
    }

    maze::TerminationReason reason = req->reason();
    if (reason != maze::TERMINATION_REASON_CLIENT_ABORT &&
        reason != maze::TERMINATION_REASON_CHAIN_FAILURE) {
        reason = maze::TERMINATION_REASON_CHAIN_FAILURE;
    }
    for (const auto& item : session->agents) {
        QuarantineAgentSamples(*session, item.first);
    }
    session->episode_state = SessionManager::EpisodeState::Aborted;
    session->episode_history[req->episode_id()] =
        SessionManager::EpisodeState::Aborted;

    rsp->set_ret_code(0);
    rsp->set_result(maze::LIFECYCLE_RESULT_OK);
    rsp->set_message("episode aborted");
    rsp->set_episode_id(req->episode_id());
    return grpc::Status::OK;
}

int64_t MazeServiceImpl::CountCachedSamples() {
    auto* session = session_mgr_.GetSession(0);
    if (!session) return 0;
    int64_t count = 0;
    for (const auto& item : session->agent_sample_caches) {
        count += static_cast<int64_t>(item.second.size());
    }
    return count;
}

int64_t MazeServiceImpl::CountCachedFragments() {
    auto* session = session_mgr_.GetSession(0);
    if (!session) return 0;
    int64_t count = 0;
    for (const auto& item : session->agent_sample_caches) {
        if (!item.second.empty()) ++count;
    }
    return count;
}

int64_t MazeServiceImpl::EstimateCachedBytes() {
    auto* session = session_mgr_.GetSession(0);
    if (!session) return 0;
    int64_t bytes = 0;
    for (const auto& item : session->pending_sample_batches) {
        bytes += static_cast<int64_t>(item.second.ByteSizeLong());
    }
    for (const auto& item : session->agent_sample_caches) {
        if (session->pending_sample_batches.find(item.first) !=
            session->pending_sample_batches.end()) {
            continue;
        }
        for (const auto& sample : item.second) {
            bytes += static_cast<int64_t>(sample.ByteSizeLong());
        }
    }
    return bytes;
}

grpc::Status MazeServiceImpl::GetAIServerStatus(
    grpc::ServerContext*,
    const maze::AIServerStatusReq* req,
    maze::AIServerStatusRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!req->run_id().empty() &&
        req->run_id() != config_.sample_output.run_id) {
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            "run_id does not match AIServer");
    }

    auto sender = sample_sender_.GetSnapshot();
    maze::AIServerState state = state_.load();
    if (sender.degraded &&
        state != maze::AISERVER_STATE_DRAINING &&
        state != maze::AISERVER_STATE_STOPPED) {
        state = maze::AISERVER_STATE_DEGRADED;
    }
    int64_t cached_samples = CountCachedSamples();
    int64_t cached_fragments = CountCachedFragments();

    rsp->set_protocol_version(2);
    rsp->set_run_id(config_.sample_output.run_id);
    rsp->set_aiserver_id(config_.sample_output.aiserver_id);
    rsp->set_producer_instance_id(producer_instance_id_);
    rsp->set_state(state);
    rsp->set_ready(
        state == maze::AISERVER_STATE_READY &&
        (config_.server.run_mode == aiserver_mode::kAstarTest ||
         model_state_.load() == maze::MODEL_STATE_READY) &&
        (config_.server.run_mode != aiserver_mode::kTraining ||
         (sender.ready && !sender.degraded)));
    rsp->set_distributor_ready(
        config_.server.run_mode == aiserver_mode::kTraining &&
        sender.ready);
    rsp->set_model_state(model_state_.load());
    rsp->set_loaded_model_version(model_manifest_.model_version);
    rsp->set_loaded_model_checksum(model_manifest_.sha256);
    rsp->set_outbound_queue_fragments(
        static_cast<int64_t>(sender.queue_fragments) + cached_fragments);
    rsp->set_outbound_queue_samples(
        sender.queue_samples + cached_samples);
    rsp->set_outbound_queue_estimated_bytes(
        sender.queue_estimated_bytes + EstimateCachedBytes());
    rsp->set_outbound_queue_high_watermark(sender.queue_high_watermark);
    rsp->set_produced_unique_samples(produced_unique_samples_);
    rsp->set_produced_unique_batches(produced_unique_batches_);
    rsp->set_push_attempt_count(sender.push_attempt_count);
    rsp->set_accepted_unique_samples(sender.accepted_unique_samples);
    rsp->set_duplicate_push_attempt_count(
        sender.duplicate_push_attempt_count);
    rsp->set_rejected_push_attempt_count(
        sender.rejected_push_attempt_count);
    rsp->set_retry_attempt_count(sender.retry_attempt_count);
    rsp->set_final_drop_unique_samples(
        sender.final_drop_unique_samples);
    rsp->set_active_session_count(
        session_mgr_.GetActiveSessionCount());
    rsp->set_active_episode_count(
        session_mgr_.GetActiveEpisodeCount());
    rsp->set_client_initialized(client_initialized_);
    rsp->set_update_rpc_count(update_rpc_count_);
    rsp->set_update_rpc_latency_sum_ms(update_rpc_latency_sum_ms_);
    rsp->set_update_rpc_latency_max_ms(update_rpc_latency_max_ms_);
    rsp->set_inference_count(inference_count_);
    rsp->set_inference_latency_sum_ms(inference_latency_sum_ms_);
    rsp->set_inference_latency_max_ms(inference_latency_max_ms_);
    rsp->set_enqueue_count(enqueue_count_);
    rsp->set_enqueue_latency_sum_ms(enqueue_latency_sum_ms_);
    rsp->set_enqueue_latency_max_ms(enqueue_latency_max_ms_);
    rsp->set_push_rpc_count(sender.push_rpc_count);
    rsp->set_push_rpc_latency_sum_ms(sender.push_rpc_latency_sum_ms);
    rsp->set_push_rpc_latency_max_ms(sender.push_rpc_latency_max_ms);
    rsp->set_last_error(
        !last_error_.empty() ? last_error_ : sender.last_error);
    rsp->set_timestamp_ms(NowMs());
    rsp->set_staged_model_version(
        staged_model_manifest_.model_version);
    rsp->set_staged_model_checksum(
        staged_model_manifest_.sha256);
    rsp->set_model_switch_count(model_switch_count_);
    rsp->set_quarantined_sample_count(
        quarantined_sample_count_);
    rsp->set_quarantined_fragment_count(
        quarantined_fragment_count_);
    return grpc::Status::OK;
}
