#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "config/run_mode.h"
#include "task/single_map_task_controller.h"

inline constexpr char kPublishedModelFile[] = "SaveModel.onnx";
inline constexpr char kModelManifestFile[] = "manifest.pb";

// ---- 服务参数 ----
struct ServerConfig {
    int listen_port = 9002;             // gRPC 监听端口
    int max_agents  = 10;               // 最大 Agent 数量
    int run_mode    = aiserver_mode::kEvaluation;
};

// ---- 策略参数 ----
struct StrategyConfig {
    int         grid_size  = 500;       // 网格大小 (cm)
    int         replan_interval = 10;   // A* 模式下重新规划间隔（帧）
};

// ---- 模型参数 ----
struct ModelConfig {
    std::string evaluation_model_path =
        "../models/eval/0000000/SaveModel.onnx";
    std::string local_train_dir = "../models/train";
    int         startup_timeout_ms = 30000;
    int         expected_obs_dim = 0;
    int         expected_action_dim = 0;
};

struct PolicyConfig {
    double training_temperature = 0.0;
    uint32_t sampling_seed = 0;
    std::string action_mask_mode;
};

struct RolloutConfig {
    double gamma = 0.99;
    double gae_lambda = 0.95;
    uint32_t tmax = 128;
};

struct ObservationConfig {
    int ray_max_range = 10;
};

struct ModelDistributionConfig {
    std::string host = "maze-learner";
    int port = 9200;
    int poll_interval_ms = 200;
    int rpc_timeout_ms = 5000;
};

// Runtime Environment assignment owned only by AIServer.
struct EnvironmentConfig {
    int agent_count = 4;
};

// Maze task ownership belongs to AIServer. Client configuration cannot
// override any value in this structure.
struct MazeTaskConfig {
    std::string fixed_map_id = "maze_117436372";
    int episode_max_steps = 1504;
};

// ---- Learner Pod 样本接入服务连接参数 ----
struct SampleDistributorConfig {
    bool        enabled           = true;
    std::string host              = "maze-learner";
    int         port              = 9100;
    int         envelope_max_transitions = 128;
    std::size_t envelope_max_bytes = 8ULL * 1024ULL * 1024ULL;
    int         rpc_timeout_ms    = 2000;
    int         max_attempts      = 4;
    int         enqueue_timeout_ms = 100;
    int         drain_timeout_ms  = 10000;
    int         health_timeout_ms = 5000;
    int         status_poll_interval_ms = 200;
    // Maximum time allowed for SamplePool ingress recovery before training
    // fails closed. The default is explicit and may be overridden by config.
    int         recovery_timeout_ms = 30000;
    std::size_t outbound_max_envelopes = 64;
    std::size_t outbound_max_estimated_bytes = 64ULL * 1024ULL * 1024ULL;
    std::string aiserver_id       = "aiserver-0";
    std::string env_id            = "env-0";
};

// ---- AIServer 完整配置 ----
struct AIServerConfig {
    int reward_metric_interval_ms = 5000;
    ServerConfig   server;
    StrategyConfig strategy;
    PolicyConfig policy;
    RolloutConfig rollout;
    ObservationConfig observation;
    ModelConfig    model;
    ModelDistributionConfig model_distribution;
    EnvironmentConfig environment;
    MazeTaskConfig task;
    SampleDistributorConfig sample_distributor;
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

// ---- 配置加载器 ----
// Load and validate runtime configuration at its owning type boundary.
bool LoadServerConfig(const std::string& yaml_path, AIServerConfig& out_config);
bool LoadServerConfig(const std::string& yaml_path,
                      const AIServerConfigOverrides& overrides,
                      AIServerConfig& out_config,
                      AIServerConfigLoadReport& report,
                      std::string& error);
