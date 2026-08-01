#pragma once

#include <string>

namespace aiserver_mode {

constexpr int kTraining = 1;
constexpr int kLocalTest = 2;
constexpr int kModelEvaluation = 3;
constexpr int kAstarTest = 4;

inline int Parse(const std::string& value) {
    if (value == "1" || value == "train" || value == "training") {
        return kTraining;
    }
    if (value == "2" || value == "local-test" ||
        value == "inference-smoke") {
        return kLocalTest;
    }
    if (value == "3" || value == "model-evaluation") {
        return kModelEvaluation;
    }
    if (value == "4" || value == "astar-test") {
        return kAstarTest;
    }
    return 0;
}

inline const char* Workload(int mode) {
    switch (mode) {
        case kTraining:
            return "training";
        case kLocalTest:
            return "local-test";
        case kModelEvaluation:
            return "model-evaluation";
        case kAstarTest:
            return "astar-test";
        default:
            return "unknown";
    }
}

inline bool IsValid(int mode) {
    return mode >= kTraining && mode <= kAstarTest;
}

}  // namespace aiserver_mode
