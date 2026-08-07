#include "grpc/maze_service.h"

#include "ai/maze_observation.h"
#include "log/logger.h"
#include "model/model_boundary.h"
#include "task/maze_map_contract.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace {

constexpr std::uint32_t kSessionProtocolVersion = 3;
constexpr const char* kActionRuleId =
    "maze.action.9-way.no-corner-cut.v1";

constexpr int kActionDirections[9][2] = {
    {0, 0}, {0, 1}, {1, 1}, {1, 0}, {1, -1},
    {0, -1}, {-1, -1}, {-1, 0}, {-1, 1},
};

double ElapsedMs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

bool IsLowerSha256(const std::string& value) {
    if (value.size() != 64) return false;
    return std::all_of(value.begin(), value.end(), [](char character) {
        return (character >= '0' && character <= '9') ||
               (character >= 'a' && character <= 'f');
    });
}

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

void FillServiceIdentity(const std::string& component,
                         const std::string& instance_id,
                         std::uint64_t lifecycle_epoch,
                         common::ServiceInstanceIdentity* target) {
    target->set_component(component);
    target->set_instance_id(instance_id);
    target->set_lifecycle_epoch(lifecycle_epoch);
}

void FillBehaviorPolicy(const AIServerConfig& config,
                        const ModelManifest& manifest,
                        maze::BehaviorPolicyBinding* target) {
    target->set_model_lineage_id(manifest.model_lineage_id);
    target->set_model_version(
        static_cast<std::uint64_t>(manifest.model_version));
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
    return run_mode == aiserver_mode::kLocalTest ||
                   run_mode == aiserver_mode::kModelEvaluation
               ? maze::REPLAY_POLICY_RECORD_AND_SERVE
               : maze::REPLAY_POLICY_DISABLED;
}

bool SameTask(const maze::TaskIdentity& lhs,
              const maze::TaskIdentity& rhs) {
    return lhs.SerializeAsString() == rhs.SerializeAsString();
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

bool IsEnvironmentTerminal(maze::MazeTerminationReason reason) {
    return reason == maze::MAZE_TERMINATION_REASON_GOAL_REACHED ||
           reason == maze::MAZE_TERMINATION_REASON_TIME_LIMIT;
}

bool ExpectedClientPosition(const SessionManager::Session& session,
                            int from_gx,
                            int from_gy,
                            int action,
                            int& expected_gx,
                            int& expected_gy) {
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

void FillLifecycle(const SessionManager::Session& session,
                   std::uint64_t applied_sequence,
                   maze::LifecycleResult result,
                   maze::LifecycleErrorCode error_code,
                   const std::string& message,
                   maze::LifecycleReply* reply) {
    reply->set_ret_code(
        result == maze::LIFECYCLE_RESULT_REJECTED ? -1 : 0);
    reply->set_result(result);
    reply->set_error_code(error_code);
    reply->set_message(message);
    reply->set_applied_sequence(applied_sequence);
    reply->set_task_state(session.task_state);
    reply->set_session_state(session.session_state);
    reply->set_episode_state(session.protocol_episode_state);
    reply->set_evaluation_state(session.evaluation_state);
}

void FillOpenRejected(maze::LifecycleErrorCode error_code,
                      const std::string& message,
                      maze::LifecycleReply* reply) {
    reply->set_ret_code(-1);
    reply->set_result(maze::LIFECYCLE_RESULT_REJECTED);
    reply->set_error_code(error_code);
    reply->set_message(message);
    reply->set_applied_sequence(0);
    reply->set_task_state(maze::TASK_STATE_CREATED);
    reply->set_session_state(maze::SESSION_STATE_OPENED);
    reply->set_episode_state(maze::EPISODE_STATE_UNSPECIFIED);
    reply->set_evaluation_state(maze::EVALUATION_STATE_INACTIVE);
}

enum class CommandCheck {
    Proceed,
    Replayed,
    Rejected,
};

template <typename Request, typename Response>
CommandCheck CheckCommand(SessionManager::Session& session,
                          const maze::LifecycleCommand& command,
                          const Request& request,
                          Response* response) {
    auto* reply = response->mutable_lifecycle();
    if (command.idempotency_key().empty()) {
        FillLifecycle(session, session.last_command_sequence,
                      maze::LIFECYCLE_RESULT_REJECTED,
                      maze::LIFECYCLE_ERROR_CODE_INVALID_IDENTITY,
                      "idempotency key is required", reply);
        return CommandCheck::Rejected;
    }
    const std::string payload = request.SerializeAsString();
    const auto existing = session.command_payloads.find(
        command.idempotency_key());
    if (existing != session.command_payloads.end()) {
        if (existing->second != payload) {
            FillLifecycle(session, session.last_command_sequence,
                          maze::LIFECYCLE_RESULT_REJECTED,
                          maze::LIFECYCLE_ERROR_CODE_IDEMPOTENCY_CONFLICT,
                          "idempotency key was reused with a different payload",
                          reply);
            return CommandCheck::Rejected;
        }
        const auto cached = session.command_responses.find(
            command.idempotency_key());
        if (cached == session.command_responses.end() ||
            !response->ParseFromString(cached->second)) {
            FillLifecycle(session, session.last_command_sequence,
                          maze::LIFECYCLE_RESULT_REJECTED,
                          maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                          "idempotent response is unavailable", reply);
            return CommandCheck::Rejected;
        }
        response->mutable_lifecycle()->set_result(
            maze::LIFECYCLE_RESULT_ALREADY_APPLIED);
        response->mutable_lifecycle()->set_message(
            "command was already applied");
        return CommandCheck::Replayed;
    }
    if (!SameTask(command.task(), session.task) ||
        command.session_id() != session.session_id) {
        FillLifecycle(session, session.last_command_sequence,
                      maze::LIFECYCLE_RESULT_REJECTED,
                      maze::LIFECYCLE_ERROR_CODE_INVALID_IDENTITY,
                      "task or session identity does not match", reply);
        return CommandCheck::Rejected;
    }
    if (command.lifecycle_epoch() != session.lifecycle_epoch) {
        FillLifecycle(session, session.last_command_sequence,
                      maze::LIFECYCLE_RESULT_REJECTED,
                      maze::LIFECYCLE_ERROR_CODE_STALE_EPOCH,
                      "lifecycle epoch does not match", reply);
        return CommandCheck::Rejected;
    }
    if (command.command_sequence() != session.last_command_sequence + 1) {
        FillLifecycle(session, session.last_command_sequence,
                      maze::LIFECYCLE_RESULT_REJECTED,
                      maze::LIFECYCLE_ERROR_CODE_OUT_OF_ORDER,
                      "command sequence is not contiguous", reply);
        return CommandCheck::Rejected;
    }
    if (command.expected_task_state() != session.task_state ||
        command.expected_session_state() != session.session_state ||
        command.expected_episode_state() != session.protocol_episode_state ||
        command.expected_evaluation_state() != session.evaluation_state) {
        FillLifecycle(session, session.last_command_sequence,
                      maze::LIFECYCLE_RESULT_REJECTED,
                      maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                      "expected lifecycle state does not match", reply);
        return CommandCheck::Rejected;
    }
    return CommandCheck::Proceed;
}

template <typename Request, typename Response>
void CommitCommand(SessionManager::Session& session,
                   const maze::LifecycleCommand& command,
                   const Request& request,
                   Response* response,
                   const std::string& message) {
    session.last_command_sequence = command.command_sequence();
    FillLifecycle(session, command.command_sequence(),
                  maze::LIFECYCLE_RESULT_APPLIED,
                  maze::LIFECYCLE_ERROR_CODE_UNSPECIFIED,
                  message, response->mutable_lifecycle());
    session.command_payloads[command.idempotency_key()] =
        request.SerializeAsString();
    session.command_responses[command.idempotency_key()] =
        response->SerializeAsString();
}

void RejectCommand(const SessionManager::Session& session,
                   maze::LifecycleErrorCode error_code,
                   const std::string& message,
                   maze::LifecycleReply* reply) {
    FillLifecycle(session, session.last_command_sequence,
                  maze::LIFECYCLE_RESULT_REJECTED, error_code,
                  message, reply);
}

void AddMetricDescriptor(training::MetricSnapshot* snapshot,
                         const std::string& field_id,
                         const std::string& label,
                         const std::string& group,
                         const std::string& dimension,
                         const std::string& unit,
                         const std::string& statistic,
                         training::MetricValueKind value_kind,
                         training::MetricAggregationKind aggregation,
                         training::MetricWindowKind window) {
    auto* descriptor = snapshot->add_descriptors();
    descriptor->set_field_id(field_id);
    descriptor->set_label(label);
    descriptor->set_group(group);
    descriptor->set_dimension(dimension);
    descriptor->set_unit(unit);
    descriptor->set_scope("server_pod");
    descriptor->set_statistic(statistic);
    descriptor->set_value_kind(value_kind);
    descriptor->set_owner_component("rl-aiserver");
    descriptor->set_aggregation_kind(aggregation);
    descriptor->set_window_kind(window);
    descriptor->mutable_schema_identity()->set_schema_id("maze.metrics.v1");
    descriptor->mutable_schema_identity()->set_schema_version(1);
    descriptor->mutable_schema_identity()->mutable_canonical_digest()->
        set_algorithm(common::DIGEST_ALGORITHM_SHA256);
    descriptor->mutable_schema_identity()->mutable_canonical_digest()->set_hex(
        "2ab9434f8c80b4651b0f51f65f3e94e29b0dd0a55803ed4c7d888f44f4604ce4");
}

void AddMetricValue(training::MetricSnapshot* snapshot,
                    const std::string& field_id,
                    double value,
                    int64_t timestamp) {
    auto* metric = snapshot->add_values();
    metric->set_field_id(field_id);
    metric->set_value(value);
    metric->set_window_end_unix_ms(timestamp);
}

void AddMetricMeanValue(training::MetricSnapshot* snapshot,
                        const std::string& field_id,
                        double sum,
                        uint64_t count,
                        int64_t timestamp) {
    auto* metric = snapshot->add_values();
    metric->set_field_id(field_id);
    metric->set_sum(sum);
    metric->set_count(count);
    metric->set_value(count > 0 ? sum / static_cast<double>(count) : 0.0);
    metric->set_window_end_unix_ms(timestamp);
}

}  // namespace

void MazeServiceImpl::RecordUpdateLatency(
    std::chrono::steady_clock::time_point start) {
    const double latency_ms = ElapsedMs(start);
    ++update_rpc_count_;
    update_rpc_latency_sum_ms_ += latency_ms;
    update_rpc_latency_max_ms_ =
        std::max(update_rpc_latency_max_ms_, latency_ms);
}

void MazeServiceImpl::MarkDegraded(const std::string& error) {
    state_.store(training::AISERVER_STATE_DEGRADED);
    last_error_ = error;
    if (config_.server.run_mode == aiserver_mode::kTraining) {
        sample_sender_.MarkDegraded(error);
    }
}

grpc::Status MazeServiceImpl::OpenSession(
    grpc::ServerContext*,
    const maze::OpenSessionReq* req,
    maze::OpenSessionRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string payload = req->SerializeAsString();
    const auto prior = open_payloads_.find(req->idempotency_key());
    if (prior != open_payloads_.end()) {
        if (prior->second != payload) {
            FillOpenRejected(
                maze::LIFECYCLE_ERROR_CODE_IDEMPOTENCY_CONFLICT,
                "OpenSession idempotency key conflicts", rsp->mutable_lifecycle());
            return grpc::Status::OK;
        }
        const auto cached = open_responses_.find(req->idempotency_key());
        if (cached != open_responses_.end() &&
            rsp->ParseFromString(cached->second)) {
            rsp->mutable_lifecycle()->set_result(
                maze::LIFECYCLE_RESULT_ALREADY_APPLIED);
            rsp->mutable_lifecycle()->set_message(
                "session was already opened");
            return grpc::Status::OK;
        }
        FillOpenRejected(maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                         "OpenSession response is unavailable",
                         rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }

    if (!IsReady()) {
        FillOpenRejected(maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                         "AIServer is not ready", rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }
    if (req->idempotency_key().empty() ||
        req->client().component() != "maze-client" ||
        req->client().instance_id().empty() ||
        req->client().lifecycle_epoch() == 0 ||
        req->environment_instance_id().empty()) {
        FillOpenRejected(maze::LIFECYCLE_ERROR_CODE_INVALID_IDENTITY,
                         "Client or environment identity is invalid",
                         rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }
    if (std::find(req->supported_session_protocol_versions().begin(),
                  req->supported_session_protocol_versions().end(),
                  kSessionProtocolVersion) ==
            req->supported_session_protocol_versions().end() ||
        !ContainsSchema(req->supported_observation_schemas(),
                        config_.training_semantics.observation_schema) ||
        !ContainsSchema(req->supported_action_schemas(),
                        config_.training_semantics.action_schema)) {
        FillOpenRejected(maze::LIFECYCLE_ERROR_CODE_UNSUPPORTED_SCHEMA,
                         "Client does not support the assigned protocol schemas",
                         rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }

    const std::string session_id = session_mgr_.CreateSession();
    auto* session = session_mgr_.GetSession(session_id);
    if (!session) {
        FillOpenRejected(maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                         "AIServer could not allocate a session",
                         rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }
    session->client.CopyFrom(req->client());
    session->environment_instance_id = req->environment_instance_id();
    session->lifecycle_epoch = next_lifecycle_epoch_.fetch_add(1);
    FillTaskIdentity(config_, &session->task);
    session->map_id = config_.task.fixed_map_id;
    session->workload_mode = WorkloadModeForRunMode(config_.server.run_mode);
    session->opened = true;
    session->task_state = maze::TASK_STATE_INITIALIZING;
    session->session_state = maze::SESSION_STATE_OPENED;
    session->protocol_episode_state = maze::EPISODE_STATE_UNSPECIFIED;
    session->evaluation_state = maze::EVALUATION_STATE_INACTIVE;

    rsp->set_session_protocol_version(kSessionProtocolVersion);
    rsp->set_session_id(session_id);
    rsp->set_lifecycle_epoch(session->lifecycle_epoch);
    FillServiceIdentity("rl-aiserver", producer_instance_id_, 1,
                        rsp->mutable_aiserver());
    auto* spec = rsp->mutable_task_spec();
    spec->mutable_identity()->CopyFrom(session->task);
    spec->set_agent_count(static_cast<std::uint32_t>(config_.task.agent_num));
    spec->set_fixed_map_id(config_.task.fixed_map_id);
    spec->mutable_expected_map_digest()->CopyFrom(
        session->task.fixed_map_digest());
    FillSchema(config_.training_semantics.observation_schema,
               spec->mutable_observation_schema());
    FillSchema(config_.training_semantics.action_schema,
               spec->mutable_action_schema());
    spec->set_action_rule_id(config_.task.action_rule_id);
    rsp->set_workload_mode(session->workload_mode);
    rsp->set_replay_policy(ReplayPolicyForRunMode(config_.server.run_mode));
    FillLifecycle(*session, 0, maze::LIFECYCLE_RESULT_APPLIED,
                  maze::LIFECYCLE_ERROR_CODE_UNSPECIFIED,
                  "session opened and task assigned", rsp->mutable_lifecycle());
    open_payloads_[req->idempotency_key()] = payload;
    open_responses_[req->idempotency_key()] = rsp->SerializeAsString();
    LOG_INFO("MazeService", "Session 已分配: %s task=%s",
             session_id.c_str(), config_.task.task_id.c_str());
    return grpc::Status::OK;
}

grpc::Status MazeServiceImpl::Init(
    grpc::ServerContext*,
    const maze::InitReq* req,
    maze::InitRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* session = session_mgr_.GetSession(req->command().session_id());
    if (!session) {
        FillOpenRejected(maze::LIFECYCLE_ERROR_CODE_INVALID_IDENTITY,
                         "session does not exist", rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }
    const auto check = CheckCommand(*session, req->command(), *req, rsp);
    if (check != CommandCheck::Proceed) return grpc::Status::OK;
    if (!session->opened || session->initialized ||
        session->task_state != maze::TASK_STATE_INITIALIZING ||
        session->session_state != maze::SESSION_STATE_OPENED ||
        !req->command().episode_id().empty() ||
        !req->command().evaluation_id().empty()) {
        RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                      "Init is not valid in the current session state",
                      rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }

    ValidatedMazeMap validated;
    std::string error;
    if (!ValidateMazeMapDescriptor(req->map(), config_.task.fixed_map_id,
                                   config_.task.fixed_map_checksum_sha256,
                                   validated, error) ||
        validated.shortest_action_steps !=
            config_.task.shortest_action_steps ||
        req->map().action_rule_id() != kActionRuleId) {
        RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_MAP_INVALID,
                      error.empty() ? "map contract identity does not match"
                                    : error,
                      rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }

    session->map_id = req->map().map_id();
    session->map_checksum_sha256 = validated.checksum_sha256;
    session->shortest_action_steps = validated.shortest_action_steps;
    session->action_rule_id = req->map().action_rule_id();
    session->grid_cols = static_cast<int>(req->map().grid_columns());
    session->grid_rows = static_cast<int>(req->map().grid_rows());
    session->grid_size_microunits = req->map().grid_size_microunits();
    session->start_gx = req->map().start_grid_x();
    session->start_gy = req->map().start_grid_y();
    session->end_gx = req->map().goal_grid_x();
    session->end_gy = req->map().goal_grid_y();
    session->blocked = std::move(validated.blocked);
    session->geodesic_distance = std::move(validated.geodesic_distance);
    session->max_finite_geodesic_distance = validated.max_finite_distance;
    session->agents.clear();
    for (int agent_id = 0; agent_id < config_.task.agent_num; ++agent_id) {
        auto& agent = session->agents[agent_id];
        if (config_.server.run_mode == aiserver_mode::kMapValidation) {
            InitAgentSolver(agent, *session);
            if (!agent.path_valid) {
                RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_MAP_INVALID,
                              "A* could not plan the validated map",
                              rsp->mutable_lifecycle());
                return grpc::Status::OK;
            }
        }
    }
    if (config_.server.run_mode == aiserver_mode::kTraining) {
        if (!task_controller_.Initialize(
                session->shortest_action_steps, ActiveModelIdentity(),
                produced_unique_samples_, error) ||
            !WriteTaskControllerReceipt(error)) {
            MarkDegraded("TaskController initialization failed: " + error);
            RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                          last_error_, rsp->mutable_lifecycle());
            return grpc::Status::OK;
        }
    }
    session->initialized = true;
    client_initialized_ = true;
    session->task_state =
        config_.server.run_mode == aiserver_mode::kTraining
            ? maze::TASK_STATE_TRAINING
            : maze::TASK_STATE_EVALUATING;
    session->session_state = maze::SESSION_STATE_IDLE;
    session->protocol_episode_state = maze::EPISODE_STATE_UNSPECIFIED;
    session->evaluation_state = maze::EVALUATION_STATE_INACTIVE;
    rsp->mutable_accepted_map_digest()->CopyFrom(
        req->map().canonical_digest());
    rsp->set_verified_shortest_action_steps(
        static_cast<std::uint32_t>(validated.shortest_action_steps));
    CommitCommand(*session, req->command(), *req, rsp,
                  "map validated and session initialized");
    return grpc::Status::OK;
}

grpc::Status MazeServiceImpl::BeginEpisode(
    grpc::ServerContext*,
    const maze::BeginEpisodeReq* req,
    maze::BeginEpisodeRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* session = session_mgr_.GetSession(req->command().session_id());
    if (!session) {
        FillOpenRejected(maze::LIFECYCLE_ERROR_CODE_INVALID_IDENTITY,
                         "session does not exist", rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }
    const auto check = CheckCommand(*session, req->command(), *req, rsp);
    if (check != CommandCheck::Proceed) return grpc::Status::OK;
    if (!session->initialized ||
        session->session_state != maze::SESSION_STATE_IDLE ||
        session->episode_state == SessionManager::EpisodeState::Active ||
        !req->command().episode_id().empty() ||
        !req->command().evaluation_id().empty()) {
        RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                      "BeginEpisode is not valid in the current state",
                      rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }

    SingleMapEpisodePlan plan;
    std::string error;
    if (task_stop_requested_) {
        plan.continue_task = false;
        plan.curriculum_stage = maze::CURRICULUM_STAGE_COMPLETE;
        plan.model = ActiveModelIdentity();
    } else if (config_.server.run_mode == aiserver_mode::kTraining) {
        const auto snapshot = task_controller_.GetSnapshot();
        if (!snapshot.evaluation_active && !snapshot.complete &&
            !snapshot.failed && !ActivateStagedModel()) {
            RejectCommand(*session,
                          maze::LIFECYCLE_ERROR_CODE_MODEL_IDENTITY_MISMATCH,
                          "staged model could not be activated",
                          rsp->mutable_lifecycle());
            return grpc::Status::OK;
        }
        if (!task_controller_.PlanNextEpisode(
                ActiveModelIdentity(), produced_unique_samples_,
                plan, error)) {
            MarkDegraded("TaskController planning failed: " + error);
            RejectCommand(*session,
                          maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                          last_error_, rsp->mutable_lifecycle());
            return grpc::Status::OK;
        }
    } else {
        plan.episode_mode =
            config_.server.run_mode == aiserver_mode::kMapValidation
                ? maze::EPISODE_MODE_EVALUATION_ARGMAX
                : maze::EPISODE_MODE_EVALUATION_ARGMAX;
        plan.curriculum_stage = maze::CURRICULUM_STAGE_8X;
        plan.max_steps = session->shortest_action_steps * 8;
        plan.model = ActiveModelIdentity();
    }

    auto* assignment = rsp->mutable_assignment();
    assignment->mutable_task()->CopyFrom(session->task);
    assignment->set_continue_task(plan.continue_task);
    assignment->set_curriculum_stage(plan.curriculum_stage);
    if (!plan.continue_task) {
        session->task_state =
            plan.curriculum_stage == maze::CURRICULUM_STAGE_FAILED
                ? maze::TASK_STATE_FAILED
                : maze::TASK_STATE_COMPLETE;
        session->session_state = maze::SESSION_STATE_IDLE;
        session->evaluation_state = maze::EVALUATION_STATE_INACTIVE;
        CommitCommand(*session, req->command(), *req, rsp,
                      "task has no further episodes");
        return grpc::Status::OK;
    }
    if (plan.max_steps <= 0 || plan.model.model_version < 0 ||
        plan.model.model_version != model_manifest_.model_version ||
        plan.model.model_checksum != model_manifest_.sha256 ||
        plan.model.trained_samples != model_manifest_.trained_samples ||
        model_manifest_.model_lineage_id.empty() ||
        !IsLowerSha256(model_manifest_.manifest_digest)) {
        RejectCommand(*session,
                      maze::LIFECYCLE_ERROR_CODE_MODEL_IDENTITY_MISMATCH,
                      "Episode plan does not match the loaded model identity",
                      rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }

    const std::string episode_id =
        "maze-episode-" + std::to_string(next_episode_id_.fetch_add(1));
    session->curriculum_stage = plan.curriculum_stage;
    session->current_episode_mode = plan.episode_mode;
    session->current_max_steps = plan.max_steps;
    const bool training_episode =
        plan.episode_mode == maze::EPISODE_MODE_TRAINING;
    session->behavior_policy_scope =
        training_episode
            ? BehaviorPolicyScope::TrainingFragment
            : BehaviorPolicyScope::EvaluationEpisode;
    session->evaluation_pinned_model_version = -1;
    session->evaluation_pinned_model_checksum.clear();
    session->evaluation_pinned_model_lineage_id.clear();
    session->evaluation_pinned_model_manifest_digest.clear();
    session->evaluation_pinned_model_trained_samples = 0;
    if (!training_episode) {
        session->evaluation_pinned_model_version = plan.model.model_version;
        session->evaluation_pinned_model_checksum = plan.model.model_checksum;
        session->evaluation_pinned_model_lineage_id =
            model_manifest_.model_lineage_id;
        session->evaluation_pinned_model_manifest_digest =
            model_manifest_.manifest_digest;
        session->evaluation_pinned_model_trained_samples =
            plan.model.trained_samples;
    }
    ResetEpisodeState(*session, episode_id);
    session->session_state = maze::SESSION_STATE_EPISODE_ACTIVE;
    session->protocol_episode_state = maze::EPISODE_STATE_RUNNING;
    session->task_state =
        plan.episode_mode == maze::EPISODE_MODE_TRAINING
            ? maze::TASK_STATE_TRAINING
            : maze::TASK_STATE_EVALUATING;
    const auto controller_snapshot = task_controller_.GetSnapshot();
    session->evaluation_state = maze::EVALUATION_STATE_INACTIVE;
    if (plan.episode_mode == maze::EPISODE_MODE_EVALUATION_STOCHASTIC) {
        session->evaluation_state =
            maze::EVALUATION_STATE_STOCHASTIC_DIAGNOSTIC;
    } else if (plan.episode_mode == maze::EPISODE_MODE_EVALUATION_ARGMAX) {
        session->evaluation_state =
            controller_snapshot.evaluation_round == 2
                ? maze::EVALUATION_STATE_ARGMAX_ROUND_2
                : maze::EVALUATION_STATE_ARGMAX_ROUND_1;
    }
    current_curriculum_stage_ = plan.curriculum_stage;
    current_episode_max_steps_ = plan.max_steps;
    latest_episode_step_ = 0;

    assignment->set_episode_id(episode_id);
    assignment->set_mode(plan.episode_mode);
    assignment->set_max_steps(static_cast<std::uint32_t>(plan.max_steps));
    assignment->set_collect_training_samples(
        plan.episode_mode == maze::EPISODE_MODE_TRAINING);
    FillBehaviorPolicy(config_, model_manifest_,
                       assignment->mutable_behavior_policy());
    if (!training_episode) {
        session->current_evaluation_id =
            "maze-evaluation-v" + std::to_string(plan.model.model_version);
        auto* evaluation = assignment->mutable_evaluation();
        evaluation->set_evaluation_id(session->current_evaluation_id);
        evaluation->set_evaluation_round(
            controller_snapshot.evaluation_round > 0
                ? static_cast<std::uint32_t>(
                      controller_snapshot.evaluation_round)
                : 1U);
        evaluation->set_state(session->evaluation_state);
        evaluation->mutable_pinned_policy()->CopyFrom(
            assignment->behavior_policy());
        evaluation->set_training_sample_emission_allowed(false);
    } else {
        session->current_evaluation_id.clear();
    }
    CommitCommand(
        *session, req->command(), *req, rsp,
        training_episode
            ? "training episode assigned with fragment-scoped behavior policy"
            : "evaluation episode assigned with episode-pinned behavior policy");
    return grpc::Status::OK;
}

grpc::Status MazeServiceImpl::Update(
    grpc::ServerContext*,
    const maze::UpdateReq* req,
    maze::UpdateRsp* rsp) {
    const auto rpc_start = std::chrono::steady_clock::now();
    rsp->set_environment_control(maze::ENVIRONMENT_CONTROL_ADVANCE);
    std::unique_lock<std::mutex> lock(mutex_);
    auto finish = [&]() { RecordUpdateLatency(rpc_start); };
    auto* session = session_mgr_.GetSession(req->command().session_id());
    if (!session) {
        FillOpenRejected(maze::LIFECYCLE_ERROR_CODE_INVALID_IDENTITY,
                         "session does not exist", rsp->mutable_lifecycle());
        finish();
        return grpc::Status::OK;
    }
    const auto check = CheckCommand(*session, req->command(), *req, rsp);
    if (check != CommandCheck::Proceed) {
        if (check == CommandCheck::Replayed) rsp->set_replayed(true);
        finish();
        return grpc::Status::OK;
    }
    if (!IsReady() || session->episode_state !=
                          SessionManager::EpisodeState::Active ||
        session->session_state != maze::SESSION_STATE_EPISODE_ACTIVE ||
        session->protocol_episode_state != maze::EPISODE_STATE_RUNNING ||
        req->command().episode_id() != session->current_episode_id ||
        req->command().evaluation_id() != session->current_evaluation_id) {
        RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                      "Update does not match the active episode",
                      rsp->mutable_lifecycle());
        finish();
        return grpc::Status::OK;
    }
    if (req->frame_id() !=
            static_cast<std::uint64_t>(session->last_frame_id + 1) ||
        req->agents_size() != static_cast<int>(session->agents.size())) {
        RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_OUT_OF_ORDER,
                      "frame is not contiguous or omits assigned Agents",
                      rsp->mutable_lifecycle());
        finish();
        return grpc::Status::OK;
    }
    const bool training_episode =
        session->current_episode_mode == maze::EPISODE_MODE_TRAINING;
    const BehaviorPolicyScope expected_policy_scope =
        training_episode
            ? BehaviorPolicyScope::TrainingFragment
            : BehaviorPolicyScope::EvaluationEpisode;
    if (session->behavior_policy_scope != expected_policy_scope) {
        RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                      "episode mode and behavior policy scope disagree",
                      rsp->mutable_lifecycle());
        finish();
        return grpc::Status::OK;
    }
    if (!training_episode &&
        (session->evaluation_pinned_model_version !=
             model_manifest_.model_version ||
         session->evaluation_pinned_model_checksum != model_manifest_.sha256 ||
         session->evaluation_pinned_model_lineage_id !=
             model_manifest_.model_lineage_id ||
         session->evaluation_pinned_model_manifest_digest !=
             model_manifest_.manifest_digest)) {
        RejectCommand(*session,
                      maze::LIFECYCLE_ERROR_CODE_MODEL_IDENTITY_MISMATCH,
                      "evaluation behavior policy identity changed",
                      rsp->mutable_lifecycle());
        finish();
        return grpc::Status::OK;
    }
    if (config_.server.run_mode == aiserver_mode::kTraining &&
        training_episode &&
        sample_sender_.IsWaitingForTrainingCapacity()) {
        rsp->set_environment_control(
            maze::ENVIRONMENT_CONTROL_WAIT_FOR_TRAINING_CAPACITY);
        rsp->set_retry_after_ms(std::max(
            1, sample_sender_.TrainingCapacityRetryAfterMs()));
        FillLifecycle(*session, session->last_command_sequence,
                      maze::LIFECYCLE_RESULT_WAIT,
                      maze::LIFECYCLE_ERROR_CODE_UNSPECIFIED,
                      "waiting for Learner sample capacity",
                      rsp->mutable_lifecycle());
        finish();
        return grpc::Status::OK;
    }

    std::unordered_set<int> seen;
    bool all_done = true;
    for (const auto& state : req->agents()) {
        const int agent_id = static_cast<int>(state.agent_id());
        auto agent_it = session->agents.find(agent_id);
        if (agent_it == session->agents.end() || !seen.insert(agent_id).second ||
            !std::isfinite(state.position().x()) ||
            !std::isfinite(state.position().y()) ||
            std::trunc(state.position().x()) != state.position().x() ||
            std::trunc(state.position().y()) != state.position().y()) {
            RejectCommand(*session,
                          maze::LIFECYCLE_ERROR_CODE_INVALID_IDENTITY,
                          "Agent identity or position is invalid",
                          rsp->mutable_lifecycle());
            finish();
            return grpc::Status::OK;
        }
        auto& agent = agent_it->second;
        const int gx = static_cast<int>(state.position().x());
        const int gy = static_cast<int>(state.position().y());
        if (!session->IsWalkable(gx, gy) ||
            state.is_done() != IsEnvironmentTerminal(
                                   state.termination_reason()) ||
            (!state.is_done() &&
             state.termination_reason() !=
                 maze::MAZE_TERMINATION_REASON_ACTIVE) ||
            (state.termination_reason() ==
                 maze::MAZE_TERMINATION_REASON_TIME_LIMIT &&
             req->frame_id() <
                 static_cast<std::uint64_t>(session->current_max_steps))) {
            RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                          "Agent state or termination reason is invalid",
                          rsp->mutable_lifecycle());
            finish();
            return grpc::Status::OK;
        }
        if (agent.has_pending_action) {
            int expected_gx = 0;
            int expected_gy = 0;
            if (!ExpectedClientPosition(*session, agent.prev_grid_x,
                                        agent.prev_grid_y,
                                        agent.pending_action,
                                        expected_gx, expected_gy) ||
                expected_gx != gx || expected_gy != gy) {
                RejectCommand(*session,
                              maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                              "Client state does not match the assigned action",
                              rsp->mutable_lifecycle());
                finish();
                return grpc::Status::OK;
            }
        } else if (agent.last_observation_frame_id >= 0 &&
                   !agent.done_collected &&
                   config_.server.run_mode == aiserver_mode::kMapValidation) {
            int expected_gx = 0;
            int expected_gy = 0;
            if (!ExpectedClientPosition(*session, agent.observation_grid_x,
                                        agent.observation_grid_y,
                                        agent.last_action,
                                        expected_gx, expected_gy) ||
                expected_gx != gx || expected_gy != gy) {
                RejectCommand(*session,
                              maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                              "Client state does not match the A* action",
                              rsp->mutable_lifecycle());
                finish();
                return grpc::Status::OK;
            }
        }
        std::string observation_error;
        if (!MazeObservation::ApplyState(
                *session, agent, gx, gy,
                static_cast<int64_t>(req->frame_id()), state.is_done(),
                state.last_move_blocked(), observation_error)) {
            RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                          observation_error, rsp->mutable_lifecycle());
            finish();
            return grpc::Status::OK;
        }
        const bool collect =
            config_.server.run_mode == aiserver_mode::kTraining &&
            session->current_episode_mode == maze::EPISODE_MODE_TRAINING &&
            !session->training_collection_paused;
        if (!agent.done_collected &&
            !FinalizePendingTransition(*session, agent_id, gx, gy,
                                       state.is_done(),
                                       state.termination_reason(), collect)) {
            RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                          last_error_, rsp->mutable_lifecycle());
            finish();
            return grpc::Status::OK;
        }
        if (state.is_done()) {
            agent.reached_goal =
                state.termination_reason() ==
                maze::MAZE_TERMINATION_REASON_GOAL_REACHED;
            agent.done_collected = true;
            agent.final_termination_reason = state.termination_reason();
        } else {
            all_done = false;
        }
    }

    if (training_sample_budget_.exceeded(produced_unique_samples_)) {
        MarkDegraded("training sample budget was exceeded");
        RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                      last_error_, rsp->mutable_lifecycle());
        finish();
        return grpc::Status::OK;
    }

    const bool collect =
        config_.server.run_mode == aiserver_mode::kTraining &&
        session->current_episode_mode == maze::EPISODE_MODE_TRAINING &&
        !session->training_collection_paused;
    const bool budget_reached =
        training_sample_budget_.reached(produced_unique_samples_);
    if (collect) {
        const bool staged_model_waiting =
            staged_model_manifest_.model_version >
            model_manifest_.model_version;
        for (const auto& state : req->agents()) {
            const int agent_id = static_cast<int>(state.agent_id());
            auto& agent = session->agents[agent_id];
            auto& cache = session->agent_sample_caches[agent_id];
            if (cache.empty()) continue;
            if (state.is_done()) {
                if (!FlushAgentSamples(*session, agent_id, true,
                                       state.termination_reason(),
                                       0.0f, false)) {
                    RejectCommand(*session,
                                  maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                                  last_error_, rsp->mutable_lifecycle());
                    finish();
                    return grpc::Status::OK;
                }
            } else if (
                budget_reached ||
                ShouldFlushAgentFragment(
                    cache.size(), current_fragment_samples_,
                    staged_model_waiting)) {
                float bootstrap_value = 0.0f;
                if (!InferStateValue(*session, agent,
                                     static_cast<int>(state.position().x()),
                                     static_cast<int>(state.position().y()),
                                     static_cast<int64_t>(req->frame_id()),
                                     bootstrap_value) ||
                    !FlushAgentSamples(*session, agent_id, false,
                                       maze::MAZE_TERMINATION_REASON_ACTIVE,
                                       bootstrap_value, true)) {
                    RejectCommand(*session,
                                  maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                                  last_error_, rsp->mutable_lifecycle());
                    finish();
                    return grpc::Status::OK;
                }
            }
        }
        if (staged_model_waiting && !ActivateStagedModel()) {
            RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                          last_error_, rsp->mutable_lifecycle());
            finish();
            return grpc::Status::OK;
        }
        bool should_pause = false;
        std::string controller_error;
        if (!task_controller_.ShouldPauseTrainingCollection(
                ActiveModelIdentity(), produced_unique_samples_,
                should_pause, controller_error)) {
            MarkDegraded("TaskController collection check failed: " +
                         controller_error);
            RejectCommand(*session,
                          maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                          last_error_, rsp->mutable_lifecycle());
            finish();
            return grpc::Status::OK;
        }
        session->training_collection_paused = should_pause;
    }

    session->last_frame_id = static_cast<int64_t>(req->frame_id());
    latest_episode_step_ = session->last_frame_id;
    if (budget_reached) {
        task_stop_requested_ = true;
        session->task_state = maze::TASK_STATE_STOPPING;
        rsp->set_task_stop_requested(true);
        rsp->set_task_stop_reason(
            maze::MAZE_TERMINATION_REASON_TASK_STOP);
        CommitCommand(*session, req->command(), *req, rsp,
                      "training sample budget reached; task stop requested");
        finish();
        return grpc::Status::OK;
    }
    if (all_done) {
        session->protocol_episode_state =
            maze::EPISODE_STATE_TERMINAL_REPORTED;
        CommitCommand(*session, req->command(), *req, rsp,
                      "terminal Agent states accepted");
        finish();
        return grpc::Status::OK;
    }

    session->last_actions.clear();
    for (const auto& state : req->agents()) {
        const int agent_id = static_cast<int>(state.agent_id());
        auto& agent = session->agents[agent_id];
        int action = 0;
        float log_probability = 0.0f;
        float value = 0.0f;
        if (!state.is_done()) {
            if (config_.server.run_mode == aiserver_mode::kMapValidation) {
                action = agent.solver.GetAction(
                    static_cast<int>(state.position().x()),
                    static_cast<int>(state.position().y()));
            } else if (!ChooseModelAction(
                           *session, agent,
                           static_cast<int>(state.position().x()),
                           static_cast<int>(state.position().y()),
                           static_cast<int64_t>(req->frame_id()),
                           action, log_probability, value)) {
                RejectCommand(*session,
                              maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                              last_error_, rsp->mutable_lifecycle());
                finish();
                return grpc::Status::OK;
            }
        }
        agent.last_action = action;
        auto* response_action = rsp->add_actions();
        response_action->set_agent_id(static_cast<std::uint32_t>(agent_id));
        response_action->set_action_id(action);
        session->last_actions.push_back(*response_action);
    }
    CommitCommand(*session, req->command(), *req, rsp,
                  "Agent states accepted and actions assigned");
    finish();
    return grpc::Status::OK;
}

grpc::Status MazeServiceImpl::EndEpisode(
    grpc::ServerContext*,
    const maze::EndEpisodeReq* req,
    maze::EndEpisodeRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* session = session_mgr_.GetSession(req->command().session_id());
    if (!session) {
        FillOpenRejected(maze::LIFECYCLE_ERROR_CODE_INVALID_IDENTITY,
                         "session does not exist", rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }
    const auto check = CheckCommand(*session, req->command(), *req, rsp);
    if (check != CommandCheck::Proceed) return grpc::Status::OK;
    if (session->episode_state != SessionManager::EpisodeState::Active ||
        session->protocol_episode_state !=
            maze::EPISODE_STATE_TERMINAL_REPORTED ||
        req->command().episode_id() != session->current_episode_id ||
        req->command().evaluation_id() != session->current_evaluation_id) {
        RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                      "EndEpisode requires a terminal report",
                      rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }

    std::vector<AgentEpisodeResult> metric_agents;
    std::vector<SingleMapAgentEvaluation> evaluation_agents;
    metric_agents.reserve(session->agents.size());
    evaluation_agents.reserve(session->agents.size());
    for (const auto& item : session->agents) {
        const auto& agent = item.second;
        if (!agent.done_collected ||
            !IsEnvironmentTerminal(agent.final_termination_reason) ||
            agent.has_pending_action ||
            !session->agent_sample_caches[item.first].empty() ||
            session->pending_sample_batches.find(item.first) !=
                session->pending_sample_batches.end()) {
            RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                          "Episode has uncommitted Agent transitions",
                          rsp->mutable_lifecycle());
            return grpc::Status::OK;
        }
        AgentEpisodeResult metric;
        metric.episode_return = agent.episode_return;
        metric.success = agent.reached_goal;
        metric.termination_reason = agent.final_termination_reason;
        metric.transition_count = agent.episode_transition_count;
        metric.shortest_action_steps = session->shortest_action_steps;
        metric.unique_cell_count = static_cast<int64_t>(agent.visited.size());
        metric.blocked_move_count = agent.blocked_move_count;
        metric.reward_component_sums = agent.reward_component_sums;
        metric_agents.push_back(std::move(metric));
        evaluation_agents.push_back(SingleMapAgentEvaluation{
            agent.reached_goal, agent.episode_transition_count});
    }
    episode_metrics_.AddCompleted(std::move(metric_agents));

    if (config_.server.run_mode == aiserver_mode::kTraining &&
        session->current_episode_mode != maze::EPISODE_MODE_TRAINING) {
        SingleMapModelIdentity pinned;
        pinned.model_version = session->evaluation_pinned_model_version;
        pinned.model_checksum = session->evaluation_pinned_model_checksum;
        pinned.trained_samples =
            session->evaluation_pinned_model_trained_samples;
        std::string error;
        if (!task_controller_.RecordEvaluationEpisode(
                session->current_episode_mode, pinned,
                evaluation_agents, error) ||
            !WriteTaskControllerReceipt(error)) {
            MarkDegraded("evaluation result commit failed: " + error);
            RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                          last_error_, rsp->mutable_lifecycle());
            return grpc::Status::OK;
        }
    }

    if (config_.server.run_mode == aiserver_mode::kLocalTest ||
        config_.server.run_mode == aiserver_mode::kModelEvaluation) {
        // Standalone evaluation is a one-shot link check. Curriculum evaluation
        // remains owned by the training TaskController and is not stopped here.
        task_stop_requested_ = true;
    }

    session->episode_state = SessionManager::EpisodeState::Ended;
    session->episode_history[session->current_episode_id] =
        SessionManager::EpisodeState::Ended;
    session->session_state = maze::SESSION_STATE_IDLE;
    session->protocol_episode_state = maze::EPISODE_STATE_COMMITTED;
    if (session->current_episode_mode == maze::EPISODE_MODE_TRAINING) {
        session->task_state = maze::TASK_STATE_TRAINING;
        session->evaluation_state = maze::EVALUATION_STATE_INACTIVE;
    } else {
        const auto snapshot = task_controller_.GetSnapshot();
        if (task_stop_requested_ || snapshot.complete) {
            session->task_state = maze::TASK_STATE_COMPLETE;
        } else if (snapshot.failed) {
            session->task_state = maze::TASK_STATE_FAILED;
        } else {
            session->task_state =
                config_.server.run_mode == aiserver_mode::kTraining
                    ? maze::TASK_STATE_TRAINING
                    : maze::TASK_STATE_EVALUATING;
        }
        session->evaluation_state = maze::EVALUATION_STATE_COMMITTED;
    }
    session->behavior_policy_scope = BehaviorPolicyScope::Unspecified;
    session->evaluation_pinned_model_version = -1;
    session->evaluation_pinned_model_checksum.clear();
    session->evaluation_pinned_model_lineage_id.clear();
    session->evaluation_pinned_model_manifest_digest.clear();
    session->evaluation_pinned_model_trained_samples = 0;
    CommitCommand(*session, req->command(), *req, rsp,
                  "Episode outcome and metrics committed");
    return grpc::Status::OK;
}

grpc::Status MazeServiceImpl::AbortEpisode(
    grpc::ServerContext*,
    const maze::AbortEpisodeReq* req,
    maze::AbortEpisodeRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* session = session_mgr_.GetSession(req->command().session_id());
    if (!session) {
        FillOpenRejected(maze::LIFECYCLE_ERROR_CODE_INVALID_IDENTITY,
                         "session does not exist", rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }
    const auto check = CheckCommand(*session, req->command(), *req, rsp);
    if (check != CommandCheck::Proceed) return grpc::Status::OK;
    const bool valid_reason =
        req->reason() == maze::MAZE_TERMINATION_REASON_CLIENT_ABORT ||
        req->reason() == maze::MAZE_TERMINATION_REASON_CHAIN_FAILURE ||
        req->reason() == maze::MAZE_TERMINATION_REASON_TASK_STOP;
    if (!valid_reason ||
        session->episode_state != SessionManager::EpisodeState::Active ||
        req->command().episode_id() != session->current_episode_id ||
        req->command().evaluation_id() != session->current_evaluation_id ||
        (req->reason() == maze::MAZE_TERMINATION_REASON_TASK_STOP &&
         !task_stop_requested_)) {
        RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                      "AbortEpisode reason or identity is invalid",
                      rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }
    for (auto& item : session->agents) {
        QuarantineAgentSamples(*session, item.first);
    }
    episode_metrics_.AddExcluded(session->agents.size(), req->reason());
    session->episode_state = SessionManager::EpisodeState::Aborted;
    session->episode_history[session->current_episode_id] =
        SessionManager::EpisodeState::Aborted;
    session->session_state = maze::SESSION_STATE_IDLE;
    session->protocol_episode_state = maze::EPISODE_STATE_ABORTED;
    session->evaluation_state = maze::EVALUATION_STATE_INACTIVE;
    session->behavior_policy_scope = BehaviorPolicyScope::Unspecified;
    session->evaluation_pinned_model_version = -1;
    session->evaluation_pinned_model_checksum.clear();
    session->evaluation_pinned_model_lineage_id.clear();
    session->evaluation_pinned_model_manifest_digest.clear();
    session->evaluation_pinned_model_trained_samples = 0;
    session->task_state =
        req->reason() == maze::MAZE_TERMINATION_REASON_TASK_STOP
            ? maze::TASK_STATE_COMPLETE
            : maze::TASK_STATE_STOPPING;
    CommitCommand(*session, req->command(), *req, rsp,
                  "Episode aborted and partial samples quarantined");
    return grpc::Status::OK;
}

grpc::Status MazeServiceImpl::CloseSession(
    grpc::ServerContext*,
    const maze::CloseSessionReq* req,
    maze::CloseSessionRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* session = session_mgr_.GetSession(req->command().session_id());
    if (!session) {
        FillOpenRejected(maze::LIFECYCLE_ERROR_CODE_INVALID_IDENTITY,
                         "session does not exist", rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }
    const auto check = CheckCommand(*session, req->command(), *req, rsp);
    if (check != CommandCheck::Proceed) return grpc::Status::OK;
    if (session->episode_state == SessionManager::EpisodeState::Active ||
        session->session_state != maze::SESSION_STATE_IDLE) {
        RejectCommand(*session, maze::LIFECYCLE_ERROR_CODE_STATE_CONFLICT,
                      "active Episode must close before its Session",
                      rsp->mutable_lifecycle());
        return grpc::Status::OK;
    }
    if (session->task_state != maze::TASK_STATE_COMPLETE &&
        session->task_state != maze::TASK_STATE_FAILED) {
        session->task_state = maze::TASK_STATE_STOPPED;
    }
    session->session_state = maze::SESSION_STATE_CLOSED;
    if (session->evaluation_state != maze::EVALUATION_STATE_COMMITTED) {
        session->evaluation_state = maze::EVALUATION_STATE_INACTIVE;
    }
    session->opened = false;
    CommitCommand(*session, req->command(), *req, rsp,
                  "Session closed without live resources");
    return grpc::Status::OK;
}

int64_t MazeServiceImpl::CountCachedSamples() {
    int64_t count = 0;
    for (const auto& session_id : session_mgr_.GetSessionIds()) {
        const auto* session = session_mgr_.GetSession(session_id);
        if (!session) continue;
        for (const auto& item : session->agent_sample_caches) {
            count += static_cast<int64_t>(item.second.size());
        }
    }
    return count;
}

int64_t MazeServiceImpl::CountCachedFragments() {
    int64_t count = 0;
    for (const auto& session_id : session_mgr_.GetSessionIds()) {
        const auto* session = session_mgr_.GetSession(session_id);
        if (!session) continue;
        for (const auto& item : session->agent_sample_caches) {
            if (!item.second.empty()) ++count;
        }
        count += static_cast<int64_t>(session->pending_sample_batches.size());
    }
    return count;
}

int64_t MazeServiceImpl::EstimateCachedBytes() {
    int64_t bytes = 0;
    for (const auto& session_id : session_mgr_.GetSessionIds()) {
        const auto* session = session_mgr_.GetSession(session_id);
        if (!session) continue;
        for (const auto& item : session->agent_sample_caches) {
            for (const auto& sample : item.second) {
                bytes += static_cast<int64_t>(sample.ByteSizeLong());
            }
        }
        for (const auto& item : session->pending_sample_batches) {
            bytes += static_cast<int64_t>(item.second.ByteSizeLong());
        }
    }
    return bytes;
}

grpc::Status MazeServiceImpl::GetAIServerStatus(
    grpc::ServerContext*,
    const training::AIServerStatusReq*,
    training::AIServerStatusRsp* rsp) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto sender = sample_sender_.GetSnapshot();
    const int64_t timestamp = NowMs();
    FillContract(config_, rsp->mutable_contract());
    FillServiceIdentity("rl-aiserver", producer_instance_id_, 1,
                        rsp->mutable_aiserver());
    rsp->set_state(state_.load());
    rsp->set_ready(IsReady());
    rsp->set_distributor_ready(
        config_.server.run_mode != aiserver_mode::kTraining || sender.ready);
    rsp->set_model_state(model_state_.load());
    if (model_manifest_.model_version >= 0) {
        rsp->mutable_loaded_model()->CopyFrom(model_manifest_.wire.identity());
    }
    if (staged_model_manifest_.model_version >= 0) {
        rsp->mutable_staged_model()->CopyFrom(
            staged_model_manifest_.wire.identity());
    }
    rsp->set_outbound_queue_fragments(
        static_cast<int64_t>(sender.queue_fragments) + CountCachedFragments());
    rsp->set_outbound_queue_samples(
        sender.queue_samples + CountCachedSamples());
    rsp->set_outbound_queue_estimated_bytes(
        sender.queue_estimated_bytes + EstimateCachedBytes());
    rsp->set_outbound_queue_high_watermark(sender.queue_high_watermark);
    rsp->set_produced_unique_samples(produced_unique_samples_);
    rsp->set_produced_unique_batches(produced_unique_batches_);
    rsp->set_push_attempt_count(sender.push_attempt_count);
    rsp->set_accepted_unique_samples(sender.accepted_unique_samples);
    rsp->set_duplicate_push_attempt_count(sender.duplicate_push_attempt_count);
    rsp->set_rejected_push_attempt_count(sender.rejected_push_attempt_count);
    rsp->set_retry_attempt_count(sender.retry_attempt_count);
    rsp->set_final_drop_unique_samples(sender.final_drop_unique_samples);
    rsp->set_active_actor_session_count(
        session_mgr_.GetActiveSessionCount());
    rsp->set_active_trajectory_count(
        session_mgr_.GetActiveEpisodeCount());
    rsp->set_inference_count(inference_count_);
    rsp->set_inference_latency_sum_ms(inference_latency_sum_ms_);
    rsp->set_inference_latency_max_ms(inference_latency_max_ms_);
    rsp->set_push_rpc_count(sender.push_rpc_count);
    rsp->set_push_rpc_latency_sum_ms(sender.push_rpc_latency_sum_ms);
    rsp->set_push_rpc_latency_max_ms(sender.push_rpc_latency_max_ms);
    rsp->set_credit_request_count(sender.credit_request_count);
    rsp->set_credit_grant_count(sender.credit_grant_count);
    rsp->set_credit_wait_count(sender.credit_wait_count);
    rsp->set_credit_reacquire_count(sender.credit_reacquire_count);
    rsp->set_producer_stale_count(sender.producer_stale_count);
    rsp->set_capacity_wait_ms(sender.capacity_wait_ms);
    rsp->set_training_capacity_wait(sender.training_capacity_wait);
    rsp->set_model_switch_count(model_switch_count_);
    rsp->set_quarantined_sample_count(quarantined_sample_count_);
    rsp->set_quarantined_fragment_count(quarantined_fragment_count_);
    rsp->set_last_error(last_error_);
    rsp->set_timestamp_unix_ms(timestamp);

    common::ServiceInstanceIdentity metric_source;
    FillServiceIdentity("rl-aiserver", producer_instance_id_, 1,
                        &metric_source);
    episode_metrics_.Fill(rsp->mutable_metrics(), metric_source,
                          next_metric_sequence_.fetch_add(1), timestamp);
    auto* metrics = rsp->mutable_metrics();
    const auto task_snapshot = task_controller_.GetSnapshot();
    AppendSingleMapTaskMetrics(
        metrics, task_snapshot, !task_controller_.history().empty(),
        timestamp);
    AddMetricDescriptor(metrics, "server.episode.max_steps.current.v1",
                        "Episode Max Steps", "episode_success", "count",
                        "step", "latest", training::METRIC_VALUE_KIND_GAUGE,
                        training::METRIC_AGGREGATION_KIND_LATEST,
                        training::METRIC_WINDOW_KIND_INSTANT);
    AddMetricValue(metrics, "server.episode.max_steps.current.v1",
                   static_cast<double>(current_episode_max_steps_), timestamp);
    AddMetricDescriptor(metrics, "server.sample.produced.total.v1",
                        "Produced Samples", "sample_flow", "count",
                        "sample", "cumulative",
                        training::METRIC_VALUE_KIND_COUNTER,
                        training::METRIC_AGGREGATION_KIND_SUM,
                        training::METRIC_WINDOW_KIND_CUMULATIVE);
    AddMetricValue(metrics, "server.sample.produced.total.v1",
                   static_cast<double>(produced_unique_samples_), timestamp);
    AddMetricDescriptor(metrics, "server.sample.accepted.total.v1",
                        "Accepted Samples", "sample_flow", "count",
                        "sample", "cumulative",
                        training::METRIC_VALUE_KIND_COUNTER,
                        training::METRIC_AGGREGATION_KIND_SUM,
                        training::METRIC_WINDOW_KIND_CUMULATIVE);
    AddMetricValue(metrics, "server.sample.accepted.total.v1",
                   static_cast<double>(sender.accepted_unique_samples),
                   timestamp);
    AddMetricDescriptor(metrics, "server.inference.latency.mean_ms.v1",
                        "Inference Latency Mean", "latency", "latency",
                        "ms", "mean", training::METRIC_VALUE_KIND_MEAN,
                        training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN,
                        training::METRIC_WINDOW_KIND_CUMULATIVE);
    AddMetricMeanValue(metrics, "server.inference.latency.mean_ms.v1",
                       inference_latency_sum_ms_, inference_count_, timestamp);
    return grpc::Status::OK;
}
