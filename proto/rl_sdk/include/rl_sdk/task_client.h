#pragma once

#include "rl_sdk/session.h"
#include "rl_sdk/transport.h"
#include "proto/common/identity.pb.h"

#include <utility>

namespace rl_sdk {

struct ClientOptions {
    rl::common::v1::ServiceInstanceIdentity identity;
    std::string environment_instance_id;
    std::string open_request_id;
    std::chrono::seconds command_timeout{5};
    std::chrono::seconds update_timeout{90};
    std::chrono::milliseconds abort_wait_budget{30000};
};

struct AcceptPayload {
    template<class Request, class Response>
    bool operator()(const Request&, const Response&) const { return true; }
};

// Protocol declares generated message/service types. Task payload validation
// stays with the caller; transport, receipts and lifecycle belong to this client.
template<class Protocol>
class TaskClient {
public:
    explicit TaskClient(ClientOptions options = {}) : options_(std::move(options)) {}
    bool Connect(const std::string& endpoint) {
        error_.clear();
        if (!transport_.Connect(endpoint)) {
            error_ = "transport could not connect to " + endpoint;
            return false;
        }
        stub_ = Protocol::Service::NewStub(transport_.channel());
        return true;
    }
    void Disconnect() { stub_.reset(); transport_.Disconnect(); }
    const LifecycleCursor& cursor() const { return session_.cursor(); }
    const std::string& error() const { return error_; }
    bool Active() const {
        return cursor().phase == rl::session::v1::SESSION_PHASE_EPISODE_RUNNING;
    }
    bool Complete() const {
        return cursor().phase == rl::session::v1::SESSION_PHASE_TASK_COMPLETE;
    }
    bool CanClose() const {
        const auto phase = cursor().phase;
        return !cursor().session_id.empty() &&
            (phase == rl::session::v1::SESSION_PHASE_OPEN ||
             phase == rl::session::v1::SESSION_PHASE_READY ||
             phase == rl::session::v1::SESSION_PHASE_ABORTED ||
             phase == rl::session::v1::SESSION_PHASE_TASK_COMPLETE);
    }

    template<class Validate = AcceptPayload>
    CommandOutcome OpenSession(typename Protocol::OpenSessionRsp& response,
                               Validate validate = {}) {
        error_.clear();
        typename Protocol::OpenSessionReq request;
        *request.mutable_client() = options_.identity;
        request.set_environment_instance_id(options_.environment_instance_id);
        request.set_request_id(options_.open_request_id);
        if (!Invoke(request, response, &Stub::OpenSession, options_.command_timeout))
            return CommandOutcome::Unknown;
        if (!CommandAccepted(response.reply())) {
            error_ = response.reply().message();
            return CommandOutcome::Rejected;
        }
        if (response.reply().applied_sequence() != 0 ||
            response.reply().phase() != rl::session::v1::SESSION_PHASE_OPEN ||
            response.session_id().empty() || response.session_epoch() == 0 ||
            response.aiserver().component().empty() ||
            response.aiserver().instance_id().empty() ||
            response.aiserver().lifecycle_epoch() == 0 || !validate(request, response)) {
            error_ = "OpenSession identity or task assignment is invalid";
            return CommandOutcome::Unknown;
        }
        session_.Bind(response.session_id(), response.session_epoch(), response.reply());
        return CommandOutcome::Applied;
    }

    template<class Validate = AcceptPayload>
    CommandOutcome Init(typename Protocol::InitReq request, typename Protocol::InitRsp& response,
                        Validate validate = {}) {
        return Exchange(std::move(request), response, &Stub::Init, options_.command_timeout,
            [validate](const auto& req, const auto& rsp) {
                return rsp.reply().phase() == rl::session::v1::SESSION_PHASE_READY && validate(req, rsp);
            }, [](const auto&) { return 0; }, [] { return false; });
    }

    template<class Validate = AcceptPayload, class Stop = std::function<bool()>>
    CommandOutcome BeginEpisode(typename Protocol::BeginEpisodeRsp& response,
        Validate validate = {}, Stop stopped = [] { return false; }) {
        typename Protocol::BeginEpisodeReq request;
        const auto result = Exchange(std::move(request), response, &Stub::BeginEpisode,
            options_.command_timeout, [validate](const auto& req, const auto& rsp) {
                const bool assigned = rsp.reply().phase() == rl::session::v1::SESSION_PHASE_EPISODE_RUNNING &&
                    rsp.has_assignment() && !rsp.assignment().episode_id().empty();
                const bool complete = rsp.reply().phase() == rl::session::v1::SESSION_PHASE_TASK_COMPLETE && rsp.has_task_complete();
                return (assigned || complete) && validate(req, rsp);
            },
            [](const auto& r) { return r.reply().result() == rl::session::v1::COMMAND_RESULT_WAIT ? 100 : 0; },
            stopped);
        if (result == CommandOutcome::Applied && Active())
            session_.SetEpisode(response.assignment().episode_id());
        return result;
    }

    template<class Validate = AcceptPayload, class Stop = std::function<bool()>>
    CommandOutcome Update(typename Protocol::UpdateReq request, typename Protocol::UpdateRsp& response,
        Validate validate = {}, Stop stopped = [] { return false; }) {
        return Exchange(std::move(request), response, &Stub::Update, options_.update_timeout,
            [validate](const auto& req, const auto& rsp) {
                const auto phase = rsp.reply().phase();
                return (phase == rl::session::v1::SESSION_PHASE_EPISODE_RUNNING ||
                        phase == rl::session::v1::SESSION_PHASE_EPISODE_TERMINAL) && validate(req, rsp);
            }, WaitDelay<typename Protocol::UpdateRsp>, stopped);
    }

    template<class Validate = AcceptPayload>
    CommandOutcome EndEpisode(typename Protocol::EndEpisodeRsp& response, Validate validate = {}) {
        typename Protocol::EndEpisodeReq request;
        const auto result = Exchange(std::move(request), response, &Stub::EndEpisode,
            options_.command_timeout, [validate](const auto& req, const auto& rsp) {
                return rsp.reply().phase() == rl::session::v1::SESSION_PHASE_READY && validate(req, rsp);
            }, [](const auto&) { return 0; }, [] { return false; });
        if (result == CommandOutcome::Applied) session_.SetEpisode({});
        return result;
    }

    CommandOutcome AbortEpisode(typename Protocol::AbortEpisodeReq request,
                                typename Protocol::AbortEpisodeRsp& response) {
        const auto result = Exchange(std::move(request), response, &Stub::AbortEpisode,
            options_.command_timeout, [](const auto&, const auto& rsp) {
                return rsp.reply().phase() == rl::session::v1::SESSION_PHASE_ABORTED;
            }, WaitDelay<typename Protocol::AbortEpisodeRsp>,
            [] { return false; }, options_.abort_wait_budget);
        if (result == CommandOutcome::Applied) session_.SetEpisode({});
        return result;
    }

    CommandOutcome CloseSession(typename Protocol::CloseSessionRsp& response) {
        typename Protocol::CloseSessionReq request;
        return Exchange(std::move(request), response, &Stub::CloseSession,
            options_.command_timeout, [](const auto&, const auto& rsp) {
                return rsp.reply().phase() == rl::session::v1::SESSION_PHASE_CLOSED;
            }, [](const auto&) { return 0; }, [] { return false; });
    }

private:
    using Stub = typename Protocol::Service::Stub;
    template<class Response> static int64_t WaitDelay(const Response& response) {
        if (!response.has_wait()) return 0;
        return response.wait().retry_after_ms() > 0 ? response.wait().retry_after_ms() : -1;
    }
    template<class Request, class Response, class Method>
    bool Invoke(const Request& request, Response& response, Method method,
                std::chrono::seconds timeout) {
        const bool ok = transport_.InvokeUnary(response, timeout,
            [&](grpc::ClientContext& context, Response& candidate) {
                return ((*stub_).*method)(&context, request, &candidate);
            });
        if (!ok) error_ = transport_.error();
        return ok;
    }
    template<class Request, class Response, class Method, class Validate, class Wait, class Stop>
    CommandOutcome Exchange(Request request, Response& response, Method method,
        std::chrono::seconds timeout, Validate validate, Wait wait, Stop stopped,
        std::chrono::milliseconds budget = std::chrono::milliseconds::max()) {
        error_.clear();
        session_.Prepare(request);
        const auto result = session_.Exchange(request, response,
            [&](const auto& req, auto& rsp) { return Invoke(req, rsp, method, timeout); },
            validate, wait, stopped, budget);
        if (result != CommandOutcome::Applied && error_.empty())
            error_ = response.reply().message().empty() ? "command was not confirmed" : response.reply().message();
        return result;
    }
    ClientOptions options_;
    Transport transport_;
    Session session_;
    std::unique_ptr<Stub> stub_;
    std::string error_;
};
}  // namespace rl_sdk
