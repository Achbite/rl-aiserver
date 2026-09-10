#pragma once

#include "task/runtime/training_transaction.h"
#include "rl_sdk/server_command.h"
#include "log/logger.h"

namespace training_service_detail {
using rl_sdk::FillReply;
using rl_sdk::CheckCommand;
using rl_sdk::CommitCommand;
using rl_sdk::RejectCommand;
using rl_sdk::CommandCheck;
inline void FillServiceIdentity(const std::string& component,
                         const std::string& instance_id,
                         std::uint64_t lifecycle_epoch,
                         common::ServiceInstanceIdentity* target) {
    target->set_component(component);
    target->set_instance_id(instance_id);
    target->set_lifecycle_epoch(lifecycle_epoch);
}

inline PolicyMode WorkloadModeForRunMode(int run_mode) {
    switch (run_mode) {
        case aiserver_mode::kTraining:
            return PolicyMode::Training;
        case aiserver_mode::kEvaluation:
            return PolicyMode::Evaluation;
        default:
            return PolicyMode::Unspecified;
    }
}

inline void FillOpenRejected(rl::session::v1::CommandErrorCode error_code,
                      const std::string& message,
                      rl::session::v1::CommandReply* reply) {
    reply->set_result(rl::session::v1::COMMAND_RESULT_REJECTED);
    reply->set_error_code(error_code);
    reply->set_message(message);
    reply->set_applied_sequence(0);
    reply->set_phase(rl::session::v1::SESSION_PHASE_UNSPECIFIED);
}

}

// Protocol supplies generated RPC types; Task owns only environment semantics.
template<class Protocol, class Task>
class TrainingTaskService final : public Protocol::Service::Service,
                                  public training::AIServerTrainingStatusService::Service {
public:
    using RpcService = typename Protocol::Service::Service;
    using Session = typename Task::Sessions::Session;
    explicit TrainingTaskService(const typename Task::ConfigType& config)
        : runtime_(config, Task::RewardMetricPrefix()), task_adapter_(config) {
        task_adapter_.RegisterMetrics(runtime_.metric_registry_);
    }
    bool Start() { return runtime_.Start(); }
    bool BeginShutdown() { return runtime_.BeginShutdown(); }
    bool IsReady() const { return runtime_.IsReady(); }
    MetricEventService& Metrics() { return runtime_.metric_service_; }
    common::ServiceInstanceIdentity MetricSourceIdentity() const { return runtime_.MetricSourceIdentity(); }
    training::GetMetricCatalogRsp MetricCatalog() const { return runtime_.MetricCatalog(); }
    grpc::Status GetAIServerStatus(grpc::ServerContext* context,
        const training::AIServerStatusReq* request, training::AIServerStatusRsp* response) override {
        return runtime_.GetAIServerStatus(context, request, response);
    }
    grpc::Status OpenSession(
        grpc::ServerContext*,
        const typename Protocol::OpenSessionReq* req,
        typename Protocol::OpenSessionRsp* rsp) override {
        std::lock_guard<std::mutex> lock(runtime_.mutex_);
        const std::string payload = req->SerializeAsString();
        const auto prior = runtime_.open_payloads_.find(req->request_id());
        if (prior != runtime_.open_payloads_.end()) {
            if (prior->second != payload) {
                training_service_detail::FillOpenRejected(
                    rl::session::v1::COMMAND_ERROR_CODE_PAYLOAD_CONFLICT,
                    "OpenSession request_id conflicts", rsp->mutable_reply());
                return grpc::Status::OK;
            }
            const auto cached = runtime_.open_responses_.find(req->request_id());
            if (cached != runtime_.open_responses_.end() &&
                rsp->ParseFromString(cached->second)) {
                auto* session = runtime_.session_mgr_.GetSession(rsp->session_id());
                if (session != nullptr) {
                    session->last_valid_client_activity_unix_ms = runtime_.NowMs();
                }
                rsp->mutable_reply()->set_result(
                    rl::session::v1::COMMAND_RESULT_ALREADY_APPLIED);
                rsp->mutable_reply()->set_message(
                    "session was already opened");
                return grpc::Status::OK;
            }
            training_service_detail::FillOpenRejected(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                             "OpenSession response is unavailable",
                             rsp->mutable_reply());
            return grpc::Status::OK;
        }

        if (!runtime_.IsReady()) {
            training_service_detail::FillOpenRejected(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                             "AIServer is not ready", rsp->mutable_reply());
            return grpc::Status::OK;
        }
        if (req->request_id().empty() ||
            req->client().component().empty() ||
            req->client().instance_id().empty() ||
            req->client().lifecycle_epoch() == 0 ||
            req->environment_instance_id().empty()) {
            training_service_detail::FillOpenRejected(rl::session::v1::COMMAND_ERROR_CODE_INVALID_IDENTITY,
                             "Client or environment identity is invalid",
                             rsp->mutable_reply());
            return grpc::Status::OK;
        }

        const std::string session_id = runtime_.session_mgr_.CreateSession();
        auto* session = runtime_.session_mgr_.GetSession(session_id);
        if (!session) {
            training_service_detail::FillOpenRejected(rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                             "AIServer could not allocate a session",
                             rsp->mutable_reply());
            return grpc::Status::OK;
        }
        session->client.CopyFrom(req->client());
        session->environment_instance_id = req->environment_instance_id();
        session->last_valid_client_activity_unix_ms = runtime_.NowMs();
        session->session_epoch = runtime_.next_lifecycle_epoch_.fetch_add(1);
        session->workload_mode = training_service_detail::WorkloadModeForRunMode(runtime_.config_.server.run_mode);
        session->phase = rl::session::v1::SESSION_PHASE_OPEN;

        rsp->set_session_id(session_id);
        rsp->set_session_epoch(session->session_epoch);
        training_service_detail::FillServiceIdentity("rl-aiserver", runtime_.producer_instance_id_,
                            runtime_.producer_lifecycle_epoch_,
                            rsp->mutable_aiserver());
        task_adapter_.FillOpen(*session, *rsp);
        training_service_detail::FillReply(*session, 0, rl::session::v1::COMMAND_RESULT_APPLIED,
                  rl::session::v1::COMMAND_ERROR_CODE_UNSPECIFIED,
                  "session opened", rsp->mutable_reply());
        runtime_.open_payloads_[req->request_id()] = payload;
        runtime_.open_responses_[req->request_id()] = rsp->SerializeAsString();
        LOG_INFO("TaskService", "Session 已分配: %s", session_id.c_str());
        return grpc::Status::OK;
    }


    grpc::Status Init(
        grpc::ServerContext*,
        const typename Protocol::InitReq* req,
        typename Protocol::InitRsp* rsp) override {
        std::lock_guard<std::mutex> lock(runtime_.mutex_);
        auto* session = CheckedSession(*req, *rsp);
        if (!session) { return grpc::Status::OK; }
        if (session->phase != rl::session::v1::SESSION_PHASE_OPEN ||
            !req->command().episode_id().empty()) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "Init is not valid in the current session state",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }
        if (!runtime_.IsReady() || runtime_.model_ack_pending_) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "AIServer is not ready to initialize the assigned task",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }

        Session candidate = *session;
        typename Task::State candidate_task_controller = task_controller_;
        TaskError failure;
        if (!task_adapter_.InitializeTask(candidate, candidate_task_controller, *req,
                runtime_.model_manifest_, runtime_.produced_unique_transitions_, failure)) {
            RejectTask(*session, failure, rsp->mutable_reply());
            return grpc::Status::OK;
        }
        candidate.phase = rl::session::v1::SESSION_PHASE_READY;
        training_service_detail::CommitCommand(candidate, req->command(), *req, rsp,
                      "task accepted and session initialized");
        if (runtime_.config_.server.run_mode == aiserver_mode::kTraining) {
            task_controller_ = std::move(candidate_task_controller);
        }
        *session = std::move(candidate);
        runtime_.client_initialized_ = true;
        return grpc::Status::OK;
    }


    grpc::Status BeginEpisode(
        grpc::ServerContext*,
        const typename Protocol::BeginEpisodeReq* req,
        typename Protocol::BeginEpisodeRsp* rsp) override {
        std::lock_guard<std::mutex> lock(runtime_.mutex_);
        auto* session = CheckedSession(*req, *rsp);
        if (!session) { return grpc::Status::OK; }
        if (!runtime_.IsCoreInferenceReady()) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "AIServer core inference is not ready to begin an episode",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }
        if (session->phase != rl::session::v1::SESSION_PHASE_READY ||
            !req->command().episode_id().empty()) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "BeginEpisode is not valid in the current state",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }

        if (runtime_.config_.server.run_mode == aiserver_mode::kTraining) {
            const auto sender = runtime_.sample_distributor_.GetSnapshot();
            const bool recoverable_delivery_state =
                sender.delivery_state ==
                    SampleDistributor::DeliveryState::kHealthy ||
                sender.delivery_state ==
                    SampleDistributor::DeliveryState::kLocalBackpressure ||
                sender.delivery_state ==
                    SampleDistributor::DeliveryState::kTransientRetry;
            if (sender.terminal_fault || sender.degraded || !sender.ready ||
                !recoverable_delivery_state) {
                training_service_detail::RejectCommand(
                    *session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                    sender.last_error.empty()
                        ? "sample delivery cannot begin an episode"
                        : "sample delivery cannot begin an episode: " +
                              sender.last_error,
                    rsp->mutable_reply());
                return grpc::Status::OK;
            }
        }

        Session candidate = *session;
        typename Task::State candidate_task_controller = task_controller_;
        const uint64_t episode_sequence = runtime_.next_episode_id_.load();
        const bool training_episode = runtime_.config_.server.run_mode == aiserver_mode::kTraining;
        candidate.current_episode_mode = training_episode ? PolicyMode::Training : PolicyMode::Evaluation;
        candidate.behavior_policy_scope = training_episode
            ? BehaviorPolicyScope::TrainingAgentSegment : BehaviorPolicyScope::EvaluationEpisode;
        candidate.evaluation_pinned_model_lineage_id.clear();
        candidate.evaluation_pinned_model_step = 0;
        if (!training_episode) {
            candidate.evaluation_pinned_model_lineage_id = runtime_.model_manifest_.model_lineage_id();
            candidate.evaluation_pinned_model_step = runtime_.model_manifest_.model_step();
        }
        TaskError failure;
        if (!task_adapter_.AssignEpisode(candidate, candidate_task_controller, runtime_.model_manifest_,
                runtime_.produced_unique_transitions_, Task::EpisodeId(episode_sequence), *rsp, failure)) {
            RejectTask(*session, failure, rsp->mutable_reply());
            return grpc::Status::OK;
        }
        candidate.phase = rl::session::v1::SESSION_PHASE_EPISODE_RUNNING;
        training_service_detail::CommitCommand(
            candidate, req->command(), *req, rsp,
            training_episode
                ? "training episode assigned with per-Agent segment model pinning"
                : "evaluation episode assigned with episode-pinned behavior policy");
        if (runtime_.config_.server.run_mode == aiserver_mode::kTraining) {
            task_controller_ = std::move(candidate_task_controller);
        }
        *session = std::move(candidate);
        runtime_.next_episode_id_.store(episode_sequence + 1);
        return grpc::Status::OK;
    }


    grpc::Status Update(
        grpc::ServerContext*,
        const typename Protocol::UpdateReq* req,
        typename Protocol::UpdateRsp* rsp) override {
        const auto rpc_start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(runtime_.mutex_);
        auto finish = [&]() { runtime_.RecordUpdateLatency(rpc_start); };
        auto* session = CheckedSession(*req, *rsp);
        if (!session) { finish(); return grpc::Status::OK; }
        if (session->phase != rl::session::v1::SESSION_PHASE_EPISODE_RUNNING) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "Update requires a running active Episode",
                          rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }
        if (req->command().episode_id() != session->current_episode_id) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_INVALID_IDENTITY,
                          "Update Episode identity does not match",
                          rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }
        if (!runtime_.IsCoreInferenceReady()) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "AIServer core inference is not ready",
                          rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }

        const bool training_episode =
            session->current_episode_mode == PolicyMode::Training;
        const BehaviorPolicyScope expected_policy_scope =
            training_episode
                ? BehaviorPolicyScope::TrainingAgentSegment
                : BehaviorPolicyScope::EvaluationEpisode;
        if (session->behavior_policy_scope != expected_policy_scope) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "Episode mode and behavior policy scope disagree",
                          rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }
        if (!training_episode &&
            (session->evaluation_pinned_model_lineage_id !=
                 runtime_.model_manifest_.model_lineage_id() ||
             session->evaluation_pinned_model_step !=
                 runtime_.model_manifest_.model_step())) {
            training_service_detail::RejectCommand(*session,
                          rl::session::v1::COMMAND_ERROR_CODE_MODEL_IDENTITY_MISMATCH,
                          "evaluation behavior policy identity changed",
                          rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }

        const bool collect =
            runtime_.config_.server.run_mode == aiserver_mode::kTraining &&
            training_episode;
        if (collect) {
            const auto sender = runtime_.sample_distributor_.GetSnapshot();
            if (sender.terminal_fault || sender.degraded) {
                training_service_detail::RejectCommand(
                    *session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                    sender.last_error.empty()
                        ? "SampleDistributor has a terminal fault"
                        : "SampleDistributor has a terminal fault: " +
                              sender.last_error,
                    rsp->mutable_reply());
                finish();
                return grpc::Status::OK;
            }
            if (sender.sample_delivery_paused ||
                sender.delivery_state !=
                    SampleDistributor::DeliveryState::kHealthy) {
                rsp->mutable_wait()->set_retry_after_ms(
                    runtime_.sample_distributor_.PauseRetryAfterMs());
                training_service_detail::FillReply(
                    *session, session->last_command_sequence,
                    rl::session::v1::COMMAND_RESULT_WAIT,
                    rl::session::v1::COMMAND_ERROR_CODE_UNSPECIFIED,
                    sender.transient_retry
                        ? "SampleDistributor is recovering its ingress transport"
                        : "SampleDistributor local outbound queue is full",
                    rsp->mutable_reply());
                finish();
                return grpc::Status::OK;
            }
            if (!sender.ready) {
                training_service_detail::RejectCommand(*session,
                              rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                              "SampleDistributor is not ready",
                              rsp->mutable_reply());
                finish();
                return grpc::Status::OK;
            }
        }

        TrainingTransaction<typename Task::Sessions> transaction(runtime_, *session);
        typename Task::State candidate_task_controller = task_controller_;
        std::vector<AgentTaskInput> inputs;
        std::vector<ModelTaskAction> actions;
        TaskError failure;
        std::string error;
        if (!task_adapter_.DecodeFrame(transaction.session, *req, collect, inputs, failure)) {
            RejectTask(*session, failure, rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }
        if (!transaction.PrepareFrame(inputs, req->frame_id(), collect, actions, error)) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT, error, rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }
        task_adapter_.EncodeActions(transaction.session, actions, *rsp);
        if (training_episode && !task_adapter_.ObserveProgress(candidate_task_controller,
                runtime_.model_manifest_, transaction.ProducedTransitions(), failure)) {
            rsp->Clear();
            RejectTask(*session, failure, rsp->mutable_reply());
            finish();
            return grpc::Status::OK;
        }
        training_service_detail::CommitCommand(transaction.session, req->command(), *req, rsp,
            actions.empty() ? "terminal Agent states accepted" : "Agent states accepted and actions assigned");
        if (Admit(transaction, *session, *rsp, true)) {
            transaction.Commit(*session);
            task_controller_ = std::move(candidate_task_controller);
        }
        finish();
        return grpc::Status::OK;
    }


    grpc::Status EndEpisode(
        grpc::ServerContext*,
        const typename Protocol::EndEpisodeReq* req,
        typename Protocol::EndEpisodeRsp* rsp) override {
        std::lock_guard<std::mutex> lock(runtime_.mutex_);
        auto* session = CheckedSession(*req, *rsp);
        if (!session) { return grpc::Status::OK; }
        if (runtime_.shutdown_started_) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "AIServer is draining and no longer accepts EndEpisode",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }
        if (session->phase != rl::session::v1::SESSION_PHASE_EPISODE_TERMINAL ||
            req->command().episode_id() != session->current_episode_id) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "EndEpisode requires a terminal report",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }

        for (const auto& item : session->agents) {
            const auto& agent = item.second;
            if (!agent.done_collected || agent.has_pending_action || agent.segment_open ||
                    !agent.segment_id.empty() || !agent.segment_transitions.empty()) {
                training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                    "Episode has an uncommitted Agent segment", rsp->mutable_reply());
                return grpc::Status::OK;
            }
        }
        Session candidate = *session;
        if (runtime_.config_.server.run_mode == aiserver_mode::kTraining &&
            session->current_episode_mode != PolicyMode::Training) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "training sessions cannot commit evaluation episodes",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }

        candidate.phase = rl::session::v1::SESSION_PHASE_READY;
        candidate.behavior_policy_scope = BehaviorPolicyScope::Unspecified;
        candidate.evaluation_pinned_model_lineage_id.clear();
        candidate.evaluation_pinned_model_step = 0;
        training::RegisteredMetricRecord metric_fact;
        TaskError failure;
        try {
            if (!task_adapter_.PrepareOutcome(*session, *rsp, runtime_.metric_registry_, metric_fact, failure)) {
                RejectTask(*session, failure, rsp->mutable_reply());
                return grpc::Status::OK;
            }
        } catch (const std::invalid_argument& error) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                std::string("episode metric producer failed: ") + error.what(), rsp->mutable_reply());
            return grpc::Status::OK;
        }
        if (session->current_episode_mode == PolicyMode::Training) {
            const auto ended_at = std::chrono::steady_clock::now();
            const int64_t observed_at_unix_ms = runtime_.NowMs();
            try {
                session->reward_metrics.AppendTo(metric_fact, runtime_.metric_registry_, Task::RewardMetricPrefix(), observed_at_unix_ms, ended_at);
            } catch (const std::invalid_argument& error) {
                training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                              std::string("episode metric producer failed: ") + error.what(),
                              rsp->mutable_reply());
                return grpc::Status::OK;
            }
            std::string metric_payload;
            if (!metric_fact.SerializeToString(&metric_payload)) {
                training_service_detail::RejectCommand(*session,
                              rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                              "Episode metric payload serialization failed",
                              rsp->mutable_reply());
                return grpc::Status::OK;
            }
            const auto append_result = runtime_.metric_events_.AppendFact(
                std::move(metric_payload), observed_at_unix_ms);
            if (!append_result.applied()) {
                training_service_detail::RejectCommand(*session,
                              rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                              "Episode metric journal is already final",
                              rsp->mutable_reply());
                return grpc::Status::OK;
            }
            if (append_result.wall_clock_regressed) {
                LOG_WARN(
                    "MetricEvent",
                    "wall_clock_regression previous_observed_at_unix_ms=%lld "
                    "observed_at_unix_ms=%lld episode=%s",
                    static_cast<long long>(
                        append_result.previous_observed_at_unix_ms),
                    static_cast<long long>(observed_at_unix_ms),
                    session->current_episode_id.c_str());
            }
        }
        candidate.reward_metrics = {};
        training_service_detail::CommitCommand(candidate, req->command(), *req, rsp,
                      "Episode outcome and metrics committed");
        *session = std::move(candidate);
        return grpc::Status::OK;
    }


    grpc::Status AbortEpisode(
        grpc::ServerContext*,
        const typename Protocol::AbortEpisodeReq* req,
        typename Protocol::AbortEpisodeRsp* rsp) override {
        std::lock_guard<std::mutex> lock(runtime_.mutex_);
        auto* session = CheckedSession(*req, *rsp);
        if (!session) { return grpc::Status::OK; }
        if (runtime_.shutdown_started_) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "AIServer is draining and no longer accepts AbortEpisode",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }
        const bool valid_reason = task_adapter_.ValidAbort(*req);
        if (!valid_reason ||
            session->phase != rl::session::v1::SESSION_PHASE_EPISODE_RUNNING ||
            req->command().episode_id() != session->current_episode_id) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "AbortEpisode reason or identity is invalid",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }

        TrainingTransaction<typename Task::Sessions> transaction(runtime_, *session);
        std::string error;
        const bool collect = runtime_.config_.server.run_mode == aiserver_mode::kTraining &&
            session->current_episode_mode == PolicyMode::Training;
        if (!transaction.PrepareAbort(collect, error)) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT, error, rsp->mutable_reply());
            return grpc::Status::OK;
        }
        training_service_detail::CommitCommand(transaction.session, req->command(), *req, rsp,
            "Episode aborted after controlled Agent segment close");
        if (Admit(transaction, *session, *rsp, false)) transaction.Commit(*session);
        return grpc::Status::OK;
    }


    grpc::Status CloseSession(
        grpc::ServerContext*,
        const typename Protocol::CloseSessionReq* req,
        typename Protocol::CloseSessionRsp* rsp) override {
        std::lock_guard<std::mutex> lock(runtime_.mutex_);
        auto* session = CheckedSession(*req, *rsp);
        if (!session) { return grpc::Status::OK; }
        const bool closeable =
            session->phase == rl::session::v1::SESSION_PHASE_OPEN ||
            session->phase == rl::session::v1::SESSION_PHASE_READY ||
            session->phase == rl::session::v1::SESSION_PHASE_ABORTED ||
            session->phase == rl::session::v1::SESSION_PHASE_TASK_COMPLETE;
        if (!closeable) {
            training_service_detail::RejectCommand(*session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "Session close requires no active Episode",
                          rsp->mutable_reply());
            return grpc::Status::OK;
        }
        session->phase = rl::session::v1::SESSION_PHASE_CLOSED;
        training_service_detail::CommitCommand(*session, req->command(), *req, rsp,
                      "Session closed without live resources");
        return grpc::Status::OK;
    }

private:
    template<class Request, class Response>
    Session* CheckedSession(const Request& request, Response& response) {
        auto* session = runtime_.session_mgr_.GetSession(request.command().session_id());
        if (!session) {
            training_service_detail::FillOpenRejected(rl::session::v1::COMMAND_ERROR_CODE_INVALID_IDENTITY,
                "session does not exist", response.mutable_reply());
            return nullptr;
        }
        const auto check = rl_sdk::CheckCommand(*session, request.command(), request, &response);
        if (check == rl_sdk::CommandCheck::Proceed || check == rl_sdk::CommandCheck::Replayed)
            session->last_valid_client_activity_unix_ms = runtime_.NowMs();
        return check == rl_sdk::CommandCheck::Proceed ? session : nullptr;
    }
    void RejectTask(const Session& session, const TaskError& error, rl::session::v1::CommandReply* reply) {
        if (error.kind == TaskError::Kind::Rollout) ++runtime_.rollout_estimator_failure_count_;
        if (error.kind != TaskError::Kind::Input) runtime_.MarkDegraded(error.message);
        rl_sdk::RejectCommand(session, error.code, error.message, reply);
    }
    template<class Response>
    bool Admit(TrainingTransaction<typename Task::Sessions>& transaction, const Session& session,
               Response& response, bool record_latency) {
        std::string error;
        const auto admitted = transaction.Admit(error, record_latency);
        if (admitted == SampleDistributor::ReservationResult::kReserved) return true;
        response.Clear();
        if (admitted == SampleDistributor::ReservationResult::kRetryableUnavailable) {
            response.mutable_wait()->set_retry_after_ms(runtime_.sample_distributor_.PauseRetryAfterMs());
            rl_sdk::FillReply(session, session.last_command_sequence, rl::session::v1::COMMAND_RESULT_WAIT,
                rl::session::v1::COMMAND_ERROR_CODE_UNSPECIFIED, error, response.mutable_reply());
        } else {
            runtime_.MarkDegraded("processed envelope-set admission failed: " + error);
            rl_sdk::RejectCommand(session, rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                runtime_.last_error_, response.mutable_reply());
        }
        return false;
    }
    TrainingRuntime<typename Task::Sessions> runtime_;
    Task task_adapter_;
    typename Task::State task_controller_;
};
