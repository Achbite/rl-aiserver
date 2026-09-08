#pragma once
#include "metrics/metric_registry.h"
#include <chrono>
#include <map>
#include <string>
#include "proto/metrics/registry.pb.h"

// Part of the business candidate: discarded commands never reach this window.
struct RewardMetricWindow {
    std::chrono::steady_clock::time_point started_at;
    uint64_t count = 0;
    double total = 0.0;
    std::map<std::string, double> components;

    template <typename Components>
    void Observe(double value, const Components& values,
                 std::chrono::steady_clock::time_point at = std::chrono::steady_clock::now()) {
        if (count == 0) started_at = at;
        ++count;
        total += value;
        for (const auto& item : values) components[item.first] += item.second;
    }
    void AppendTo(training::RegisteredMetricRecord& record, const MetricRegistry& registry,
                  const std::string& prefix, int64_t end_unix_ms,
                  std::chrono::steady_clock::time_point ended_at = std::chrono::steady_clock::now()) const {
        if (count == 0) return;
        // Measure elapsed time monotonically, then anchor the interval to its
        // closing wall time. A wall-clock adjustment cannot reverse its duration.
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            ended_at - started_at).count();
        const int64_t start_unix_ms = end_unix_ms - elapsed_ms;
        auto add = [&](const std::string& id, double sum) {
            auto& point = registry.Mean(record, id, sum, count);
            point.set_interval_start_unix_ms(start_unix_ms);
            point.set_interval_end_unix_ms(end_unix_ms);
        };
        add(prefix + "total.per_transition", total);
        for (const auto& item : components) add(prefix + item.first + ".per_transition", item.second);
    }
};
