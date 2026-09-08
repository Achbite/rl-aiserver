#pragma once
#include "proto/communication/session.pb.h"
#include "rl_sdk/replay_window.h"
#include <cstdint>
#include <string>

namespace rl_sdk {
template<class SessionState>
void FillReply(const SessionState& session,
               std::uint64_t applied_sequence,
               rl::session::v1::CommandResult result,
               rl::session::v1::CommandErrorCode error_code,
               const std::string& message,
               rl::session::v1::CommandReply* reply) {
    reply->set_result(result);
    reply->set_error_code(error_code);
    reply->set_message(message);
    reply->set_applied_sequence(applied_sequence);
    reply->set_phase(session.phase);
}

enum class CommandCheck {
    Proceed,
    Replayed,
    Rejected,
};

template <typename Request, typename Response, typename SessionState>
CommandCheck CheckCommand(SessionState& session,
                          const rl::session::v1::CommandIdentity& command,
                          const Request& request,
                          Response* response) {
    auto* reply = response->mutable_reply();
    const std::string payload = request.SerializeAsString();
    const auto replay_decision = session.command_replay.Classify(
        command.sequence(), session.last_command_sequence, payload);
    if (replay_decision ==
        LifecycleReplayDecision::PayloadConflict) {
            FillReply(session, session.last_command_sequence,
                          rl::session::v1::COMMAND_RESULT_REJECTED,
                          rl::session::v1::COMMAND_ERROR_CODE_PAYLOAD_CONFLICT,
                          "command sequence was reused with a different payload",
                          reply);
            return CommandCheck::Rejected;
    }
    if (replay_decision == LifecycleReplayDecision::Replay) {
        if (!response->ParseFromString(session.command_replay.response())) {
            FillReply(session, session.last_command_sequence,
                          rl::session::v1::COMMAND_RESULT_REJECTED,
                          rl::session::v1::COMMAND_ERROR_CODE_STATE_CONFLICT,
                          "idempotent response is unavailable", reply);
            return CommandCheck::Rejected;
        }
        response->mutable_reply()->set_result(
            rl::session::v1::COMMAND_RESULT_ALREADY_APPLIED);
        response->mutable_reply()->set_message(
            "command was already applied");
        return CommandCheck::Replayed;
    }
    if (replay_decision == LifecycleReplayDecision::OutOfOrder) {
        FillReply(session, session.last_command_sequence,
                      rl::session::v1::COMMAND_RESULT_REJECTED,
                      rl::session::v1::COMMAND_ERROR_CODE_OUT_OF_ORDER,
                      "command sequence is not contiguous", reply);
        return CommandCheck::Rejected;
    }
    if (command.session_id() != session.session_id) {
        FillReply(session, session.last_command_sequence,
                      rl::session::v1::COMMAND_RESULT_REJECTED,
                      rl::session::v1::COMMAND_ERROR_CODE_INVALID_IDENTITY,
                      "session identity does not match", reply);
        return CommandCheck::Rejected;
    }
    if (command.session_epoch() != session.session_epoch) {
        FillReply(session, session.last_command_sequence,
                      rl::session::v1::COMMAND_RESULT_REJECTED,
                      rl::session::v1::COMMAND_ERROR_CODE_STALE_EPOCH,
                      "Session epoch does not match", reply);
        return CommandCheck::Rejected;
    }
    return CommandCheck::Proceed;
}

template <typename Request, typename Response, typename SessionState>
void CommitCommand(SessionState& session,
                   const rl::session::v1::CommandIdentity& command,
                   const Request& request,
                   Response* response,
                   const std::string& message) {
    session.last_command_sequence = command.sequence();
    FillReply(session, command.sequence(),
                  rl::session::v1::COMMAND_RESULT_APPLIED,
                  rl::session::v1::COMMAND_ERROR_CODE_UNSPECIFIED,
                  message, response->mutable_reply());
    session.command_replay.Store(
        command.sequence(), request.SerializeAsString(),
        response->SerializeAsString());
}

template<class SessionState>
void RejectCommand(const SessionState& session,
                   rl::session::v1::CommandErrorCode error_code,
                   const std::string& message,
                   rl::session::v1::CommandReply* reply) {
    FillReply(session, session.last_command_sequence,
                  rl::session::v1::COMMAND_RESULT_REJECTED, error_code,
                  message, reply);
}

}  // namespace rl_sdk
