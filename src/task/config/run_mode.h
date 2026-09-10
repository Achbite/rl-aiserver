#pragma once

#include <string>

namespace aiserver_mode {

constexpr int kTraining = 1;
constexpr int kEvaluation = 2;

inline int Parse(const std::string& value) {
    if (value == "training") {
        return kTraining;
    }
    if (value == "evaluation") {
        return kEvaluation;
    }
    return 0;
}

inline const char* Workload(int mode) {
    switch (mode) {
        case kTraining:
            return "training";
        case kEvaluation:
            return "evaluation";
        default:
            return "unknown";
    }
}

inline bool IsValid(int mode) {
    return mode == kTraining || mode == kEvaluation;
}

inline bool ExposesTrainingStatus(int mode) {
    return mode == kTraining;
}

}  // namespace aiserver_mode
