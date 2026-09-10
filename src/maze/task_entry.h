#pragma once

#include "maze/config/config.h"
#include "maze/protocol/service.h"
#include "log/logger.h"

namespace aiserver {

// Local application composition. The wire types come only from generated Proto.
struct TaskEntry {
    using Config = MazeConfig;
    using Overrides = MazeConfigOverrides;
    using LoadReport = MazeConfigLoadReport;
    using Service = MazeTaskService;

    static bool LoadConfig(const std::string& path, const Overrides& overrides,
                           Config& config, LoadReport& report, std::string& error) {
        return LoadMazeConfig(path, overrides, config, report, error);
    }

    static void LogConfig(const Config& config) {
        LOG_INFO("Maze", "agent_count=%d, map=%s", config.environment.agent_count,
                 config.task.fixed_map_id.c_str());
    }
};

}  // namespace aiserver
