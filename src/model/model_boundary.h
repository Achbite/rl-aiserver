#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

inline bool IsModelSampleBoundary(int64_t produced_samples,
                                  int64_t sample_quantum) {
    return produced_samples > 0 && sample_quantum > 0 &&
           produced_samples % sample_quantum == 0;
}

inline bool ShouldFlushAgentFragment(
    std::size_t cached_samples,
    int configured_fragment_samples,
    int64_t produced_samples,
    int64_t sample_quantum) {
    if (cached_samples == 0 || configured_fragment_samples <= 0) {
        return false;
    }
    return cached_samples >=
               static_cast<std::size_t>(configured_fragment_samples) ||
           IsModelSampleBoundary(produced_samples, sample_quantum);
}

inline int SelectModelBoundaryTarget(
    int initial_model_version,
    int64_t sample_quantum,
    const std::unordered_map<int, int64_t>& produced_samples_by_model) {
    int64_t trainable_updates = 0;
    for (const auto& item : produced_samples_by_model) {
        trainable_updates += item.second / sample_quantum;
    }
    return initial_model_version +
        static_cast<int>(trainable_updates);
}

inline int SelectPerAgentFragmentSamples(
    int64_t active_model_samples,
    int64_t sample_quantum,
    int agent_num,
    int configured_fragment_samples,
    bool all_agents_active) {
    if (!all_agents_active || active_model_samples < 0 ||
        sample_quantum <= 0 || agent_num <= 0 ||
        configured_fragment_samples <= 0) {
        return configured_fragment_samples;
    }
    const int64_t remainder = active_model_samples % sample_quantum;
    const int64_t samples_to_boundary =
        remainder == 0 ? sample_quantum : sample_quantum - remainder;
    if (samples_to_boundary % agent_num != 0) {
        return configured_fragment_samples;
    }
    const int64_t candidate = samples_to_boundary / agent_num;
    if (candidate <= 0 || candidate > configured_fragment_samples) {
        return configured_fragment_samples;
    }
    return static_cast<int>(candidate);
}
