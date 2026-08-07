#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

enum class LifecycleReplayDecision {
    Proceed,
    Replay,
    IdempotencyConflict,
    OutOfOrder,
};

// Lifecycle commands are strictly contiguous. A Client cannot submit command
// N+1 until it has accepted the reply for N, so only the most recently applied
// command is eligible for transport-level replay. Retaining every historical
// request and response makes a long-running training Session grow with every
// environment frame and does not add a valid recovery path.
class LifecycleReplayWindow {
public:
    LifecycleReplayDecision Classify(
        std::uint64_t command_sequence,
        std::uint64_t last_applied_sequence,
        const std::string& idempotency_key,
        const std::string& payload) const {
        if (present_ && idempotency_key == idempotency_key_) {
            if (payload != payload_) {
                return LifecycleReplayDecision::IdempotencyConflict;
            }
            return command_sequence == command_sequence_
                       ? LifecycleReplayDecision::Replay
                       : LifecycleReplayDecision::OutOfOrder;
        }
        if (command_sequence != last_applied_sequence + 1) {
            return LifecycleReplayDecision::OutOfOrder;
        }
        return LifecycleReplayDecision::Proceed;
    }

    void Store(std::uint64_t command_sequence,
               const std::string& idempotency_key,
               std::string payload,
               std::string response) {
        command_sequence_ = command_sequence;
        std::string(idempotency_key).swap(idempotency_key_);
        payload.swap(payload_);
        response.swap(response_);
        present_ = true;
    }

    bool present() const { return present_; }
    std::uint64_t command_sequence() const { return command_sequence_; }
    const std::string& response() const { return response_; }

    std::size_t RetainedCapacityBytes() const {
        return idempotency_key_.capacity() + payload_.capacity() +
               response_.capacity();
    }

private:
    bool present_ = false;
    std::uint64_t command_sequence_ = 0;
    std::string idempotency_key_;
    std::string payload_;
    std::string response_;
};
