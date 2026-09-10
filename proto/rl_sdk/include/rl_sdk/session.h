#pragma once
#include "proto/communication/session.pb.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

namespace rl_sdk {
struct LifecycleCursor {
    std::string session_id;
    std::string episode_id;
    std::uint64_t session_epoch = 0;
    std::uint64_t next_sequence = 1;
    rl::session::v1::SessionPhase phase = rl::session::v1::SESSION_PHASE_UNSPECIFIED;
};

inline void UpdateCursor(LifecycleCursor& cursor,
                         const rl::session::v1::CommandReply& reply) {
    cursor.phase = reply.phase();
}

// Constructing a command is side-effect free. The sequence is committed only
// after the server proves that this exact command was applied.
inline void FillCommand(const LifecycleCursor& cursor,
                        rl::session::v1::CommandIdentity* command) {
    command->set_session_id(cursor.session_id);
    command->set_session_epoch(cursor.session_epoch);
    command->set_sequence(cursor.next_sequence);
    command->set_episode_id(cursor.episode_id);
}

inline bool HasConcretePhase(const rl::session::v1::CommandReply& reply) {
    return reply.phase() >= rl::session::v1::SESSION_PHASE_OPEN &&
           reply.phase() <= rl::session::v1::SESSION_PHASE_ABORTED;
}

inline bool CommandAccepted(const rl::session::v1::CommandReply& reply) {
    return (reply.result() == rl::session::v1::COMMAND_RESULT_APPLIED ||
            reply.result() == rl::session::v1::COMMAND_RESULT_ALREADY_APPLIED) &&
           reply.error_code() == rl::session::v1::COMMAND_ERROR_CODE_UNSPECIFIED;
}

inline bool AcceptCommandReply(LifecycleCursor& cursor,
                               const rl::session::v1::CommandReply& reply) {
    const std::uint64_t expected_sequence = cursor.next_sequence;
    if (!CommandAccepted(reply) ||
        reply.applied_sequence() != expected_sequence ||
        !HasConcretePhase(reply)) {
        return false;
    }
    UpdateCursor(cursor, reply);
    cursor.next_sequence = expected_sequence + 1;
    return true;
}

// WAIT is an authoritative proof that this exact command was not applied. It
// is safe only when the server echoes the committed lifecycle cursor and the
// last applied sequence; callers must retry the unchanged command bytes.
inline bool IsCommandWait(const LifecycleCursor& cursor,
                          const rl::session::v1::CommandReply& reply) {
    return cursor.next_sequence > 0 &&
           reply.result() == rl::session::v1::COMMAND_RESULT_WAIT &&
           reply.error_code() == rl::session::v1::COMMAND_ERROR_CODE_UNSPECIFIED &&
           reply.applied_sequence() == cursor.next_sequence - 1 &&
           HasConcretePhase(reply) && reply.phase() == cursor.phase;
}

// A rejected command is safe to replace with another operation at the same
// sequence only when the server proves that it committed no lifecycle state.
inline bool IsConclusiveRejected(const LifecycleCursor& cursor,
                                 const rl::session::v1::CommandReply& reply) {
    return cursor.next_sequence > 0 &&
           reply.result() == rl::session::v1::COMMAND_RESULT_REJECTED &&
           reply.error_code() != rl::session::v1::COMMAND_ERROR_CODE_UNSPECIFIED &&
           reply.applied_sequence() == cursor.next_sequence - 1 &&
           HasConcretePhase(reply) && reply.phase() == cursor.phase;
}


enum class CommandOutcome { Applied, Rejected, Unknown, Stopped, WaitExpired };

// One cursor and one immutable request per synchronous Session. Task bindings
// validate their payload before an applied reply advances this cursor.
class Session {
public:
    const LifecycleCursor& cursor() const { return cursor_; }
    void Bind(const std::string& id, uint64_t epoch,
              const rl::session::v1::CommandReply& reply) {
        cursor_.session_id = id;
        cursor_.session_epoch = epoch;
        cursor_.next_sequence = reply.applied_sequence() + 1;
        UpdateCursor(cursor_, reply);
    }
    void SetEpisode(std::string id) { cursor_.episode_id = std::move(id); }
    template<class Request> void Prepare(Request& request) const {
        FillCommand(cursor_, request.mutable_command());
    }
    template<class Request, class Response, class Invoke, class Validate, class Wait>
    CommandOutcome Exchange(const Request& request, Response& response,
                            Invoke invoke, Validate validate, Wait wait,
                            const std::function<bool()>& stopped = [] { return false; },
                            std::chrono::milliseconds budget = std::chrono::milliseconds::max()) {
        const auto start = std::chrono::steady_clock::now();
        while (!stopped()) {
            if (budget != std::chrono::milliseconds::max() &&
                std::chrono::steady_clock::now() - start >= budget)
                return CommandOutcome::WaitExpired;
            response.Clear();
            if (!invoke(request, response)) return CommandOutcome::Unknown;
            const int64_t delay = wait(response);
            if (delay > 0 && IsCommandWait(cursor_, response.reply())) {
                auto sleep = std::chrono::milliseconds(delay);
                if (budget != std::chrono::milliseconds::max()) {
                    const auto remaining = budget - std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start);
                    sleep = std::min(sleep, std::max(remaining, std::chrono::milliseconds::zero()));
                }
                std::this_thread::sleep_for(sleep);
                continue;
            }
            if (delay == 0 && validate(request, response) &&
                AcceptCommandReply(cursor_, response.reply())) return CommandOutcome::Applied;
            return IsConclusiveRejected(cursor_, response.reply())
                ? CommandOutcome::Rejected : CommandOutcome::Unknown;
        }
        return CommandOutcome::Stopped;
    }
private:
    LifecycleCursor cursor_;
};

// Report a final environment result even when no subsequent action is needed.
// The binding supplies typed requests, receipts and environment operations.
template<class Observe, class Exchange, class Terminal, class Apply, class Stop>
CommandOutcome RunEpisode(Observe observe, Exchange exchange,
                          Terminal terminal, Apply apply, Stop stopped) {
    while (!stopped()) {
        const bool final_result = terminal();
        auto request = observe();
        const auto result = exchange(request);
        if (result != CommandOutcome::Applied) return result;
        if (final_result) return CommandOutcome::Applied;
        if (!apply()) return CommandOutcome::Rejected;
    }
    return CommandOutcome::Stopped;
}
// The binding supplies task payloads and environment effects. The SDK owns the
// Open/Init/Episode/Abort/Close order and never substitutes a command for Unknown.
template <typename Binding>
CommandOutcome RunSession(Binding& binding) {
    auto result = binding.Open();
    if (result != CommandOutcome::Applied) return result;
    auto finish = [&](CommandOutcome outcome) {
        if (outcome == CommandOutcome::Unknown) return outcome;
        if (binding.Active()) {
            const auto aborted = binding.Abort();
            if (aborted != CommandOutcome::Applied) return aborted;
        }
        if (binding.CanClose()) {
            const auto closed = binding.Close();
            if (closed != CommandOutcome::Applied) return closed;
        }
        return outcome;
    };
    result = binding.Initialize();
    if (result != CommandOutcome::Applied) return finish(result);
    while (!binding.Stopped()) {
        result = binding.Begin();
        if (result != CommandOutcome::Applied) return finish(result);
        if (binding.Complete()) return finish(CommandOutcome::Applied);
        result = binding.RunEpisode();
        if (result != CommandOutcome::Applied) return finish(result);
        result = binding.End();
        if (result != CommandOutcome::Applied) return finish(result);
    }
    return finish(CommandOutcome::Stopped);
}

}  // namespace rl_sdk
