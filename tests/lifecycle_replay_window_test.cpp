#include "session/lifecycle_replay_window.h"

#include <iostream>
#include <string>

namespace {

bool Require(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

}  // namespace

int main() {
    LifecycleReplayWindow window;
    if (!Require(
            window.Classify(1, 0, "session:1:init", "request-1") ==
                LifecycleReplayDecision::Proceed,
            "the first contiguous command must proceed")) {
        return 1;
    }

    window.Store(1, "session:1:init", "request-1", "response-1");
    if (!Require(
            window.Classify(1, 1, "session:1:init", "request-1") ==
                LifecycleReplayDecision::Replay,
            "the latest command must be replayable") ||
        !Require(window.response() == "response-1",
                 "the replay response must match the latest command") ||
        !Require(
            window.Classify(1, 1, "session:1:init", "changed") ==
                LifecycleReplayDecision::IdempotencyConflict,
            "reusing the latest key with a different payload must conflict") ||
        !Require(
            window.Classify(3, 1, "session:1:update-3", "request-3") ==
                LifecycleReplayDecision::OutOfOrder,
            "a non-contiguous command must be rejected")) {
        return 1;
    }

    const std::string large(1024 * 1024, 'x');
    window.Store(2, "session:1:update-2", large, large);
    window.Store(3, "session:1:update-3", "small-request", "small-response");
    if (!Require(
            window.Classify(2, 3, "session:1:update-2", large) ==
                LifecycleReplayDecision::OutOfOrder,
            "an older applied command must fall outside the replay window") ||
        !Require(window.RetainedCapacityBytes() < 4096,
                 "overwriting the replay window must release large buffers")) {
        return 1;
    }

    for (std::uint64_t sequence = 4; sequence <= 100000; ++sequence) {
        const std::string key = "session:1:update-" +
                                std::to_string(sequence);
        window.Store(sequence, key, "request", "response");
    }
    if (!Require(window.command_sequence() == 100000,
                 "the replay window must retain only the newest sequence") ||
        !Require(window.RetainedCapacityBytes() < 4096,
                 "long-running command traffic must keep bounded storage")) {
        return 1;
    }

    return 0;
}
