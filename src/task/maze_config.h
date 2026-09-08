#pragma once

#include "config/training_runtime_config.h"

#include <optional>
#include <string>
#include <vector>

struct ObservationConfig {
    int ray_max_range = 10;
};

// The Maze adapter owns the environment assignment sent to its Client.
struct EnvironmentConfig {
    int agent_count = 4;
};

struct MazeTaskConfig {
    std::string fixed_map_id = "maze_117436372";
    int episode_max_steps = 1504;
};

struct AIServerConfig : TrainingRuntimeConfig {
    ObservationConfig observation;
    EnvironmentConfig environment;
    MazeTaskConfig task;
};

struct AIServerConfigOverrides {
    std::optional<int> workload;
    std::optional<int> listen_port;
    std::optional<std::string> evaluation_model_path;
    std::optional<std::string> sample_distributor_host;
    std::optional<int> sample_distributor_port;
    std::optional<std::string> model_distributor_host;
    std::optional<int> model_distributor_port;
};

struct AIServerConfigLoadReport {
    std::string config_path;
    std::vector<std::string> environment_overridden_fields;
    std::vector<std::string> cli_overridden_fields;
};

// Parse the current Maze task configuration and its shared training settings.
bool LoadServerConfig(const std::string& yaml_path, AIServerConfig& out_config);
bool LoadServerConfig(const std::string& yaml_path,
                      const AIServerConfigOverrides& overrides,
                      AIServerConfig& out_config,
                      AIServerConfigLoadReport& report,
                      std::string& error);
