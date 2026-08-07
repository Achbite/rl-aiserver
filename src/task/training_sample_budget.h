#pragma once

#include <cstdint>
#include <string>

class TrainingSampleBudget {
public:
    TrainingSampleBudget(int64_t requested_samples,
                         int64_t sample_quantum)
        : requested_samples_(requested_samples),
          sample_quantum_(sample_quantum),
          effective_samples_(
              requested_samples > 0 && sample_quantum > 0
                  ? requested_samples - requested_samples % sample_quantum
                  : 0) {}

    bool Validate(std::string& error) const {
        if (requested_samples_ < 0) {
            error = "training sample budget must be non-negative";
            return false;
        }
        if (sample_quantum_ <= 0) {
            error = "training sample quantum must be positive";
            return false;
        }
        if (requested_samples_ > 0 && effective_samples_ == 0) {
            error = "training sample budget is smaller than one sample quantum";
            return false;
        }
        return true;
    }

    bool enabled() const { return effective_samples_ > 0; }
    int64_t requested_samples() const { return requested_samples_; }
    int64_t effective_samples() const { return effective_samples_; }
    int64_t sample_quantum() const { return sample_quantum_; }

    bool reached(int64_t produced_samples) const {
        return enabled() && produced_samples >= effective_samples_;
    }

    bool exceeded(int64_t produced_samples) const {
        return enabled() && produced_samples > effective_samples_;
    }

private:
    int64_t requested_samples_ = 0;
    int64_t sample_quantum_ = 0;
    int64_t effective_samples_ = 0;
};
