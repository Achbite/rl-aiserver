#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "ai/maze_reward.h"
#include "config/run_mode.h"
#include "task/single_map_task_controller.h"

inline constexpr char kLocalEvaluationModelFile[] = "SaveModel.onnx";

// ---- 服务参数 ----
struct ServerConfig {
    int listen_port = 9002;             // gRPC 监听端口
    int max_agents  = 10;               // 最大 Agent 数量
    int run_mode    = aiserver_mode::kLocalTest;
};

// ---- 策略参数 ----
struct StrategyConfig {
    int         grid_size  = 500;       // 网格大小 (cm)
    int         replan_interval = 10;   // A* 模式下重新规划间隔（帧）
};

// ---- 模型参数 ----
struct ModelConfig {
    std::string evaluation_dir = "models/local";
    std::string local_train_dir = "models/local-train";
    std::string local_test_dir = "models/local-test";
    std::string manifest_name = "manifest.json";
    int         startup_timeout_ms = 30000;
    int         expected_obs_dim = 17;
    int         expected_action_dim = 9;
    std::string observation_schema_id = "maze.observation.v3";
    std::string action_schema_id = "maze.action.v1";
    std::string model_architecture_id = "maze.mlp-17x64x64.v1";
    std::string tensor_dtype = "float32";
    std::string expected_model_lineage_id = "maze-fixed-map-seed-0";
};

struct DigestConfig {
    std::string algorithm = "sha256";
    std::string hex;
};

struct SchemaConfig {
    std::string schema_id;
    uint32_t schema_version = 1;
    DigestConfig canonical_digest;
};

struct ContractConfig {
    std::string package_name = "rl-contracts";
    std::string package_version = "0.8.0";
    DigestConfig source_digest;
    DigestConfig artifact_digest;
    std::string platform = "linux/arm64";
    std::string generator_identity;
};

struct TrainingSemanticsConfig {
    std::string training_contract_id = "maze.training.v3";
    SchemaConfig observation_schema;
    SchemaConfig action_schema;
    SchemaConfig reward_schema;
    std::string policy_distribution_schema_id =
        "categorical.logits.v1";
    std::string model_architecture_id = "maze.mlp-17x64x64.v1";
    DigestConfig semantics_digest;
};

struct PolicyConfig {
    std::string distribution_schema_id = "categorical.logits.v1";
    double training_temperature = 1.0;
    DigestConfig policy_spec_digest;
    uint32_t sampling_seed = 0;
};

struct ObservationConfig {
    int ray_max_range = 10;
};

inline std::string LocalEvaluationModelPath(const ModelConfig& config) {
    return (std::filesystem::path(config.evaluation_dir) /
            kLocalEvaluationModelFile)
        .string();
}

inline std::string LocalEvaluationManifestPath(const ModelConfig& config) {
    return (std::filesystem::path(config.evaluation_dir) /
            config.manifest_name)
        .string();
}

struct ModelDistributionConfig {
    std::string host = "maze-learner";
    int port = 9200;
    int poll_interval_ms = 200;
    int boundary_wait_ms = 60000;
    int rpc_timeout_ms = 5000;
    std::string contract_version = "0.8.0";
};

// Maze task ownership belongs to AIServer. Client configuration cannot
// override any value in this structure.
struct MazeTaskConfig {
    std::string task_contract_id = "maze.task.v3";
    std::string task_id = "maze.fixed.single-map.v1";
    uint64_t task_revision = 1;
    DigestConfig task_config_digest{
        "sha256",
        "17f885bd9ff1a20cf9fba210f1454487ea04c36251730b6d0875c4dbb5cb99d7"};
    int agent_num = 4;
    std::string fixed_map_id = "maze_117436372";
    std::string fixed_map_checksum_sha256 =
        "861e0bb22a8b9a2ed689527d080c65ec2c822367e985c49753e1be9cf3ca8ae9";
    std::string action_rule_id =
        "maze.action.9-way.no-corner-cut.v1";
    int shortest_action_steps = 188;
    // Zero means that the TaskController owns an unbounded training run.
    // Positive values are upper bounds and are rounded down to a complete
    // global sample fragment by the AIServer.
    int64_t training_sample_budget = 0;
};

// ---- Learner Pod 样本接入服务连接参数 ----
struct SampleOutputConfig {
    bool        enabled           = true;
    std::string host              = "maze-learner";
    int         port              = 9100;
    int         fragment_samples  = 128;
    int         rpc_timeout_ms    = 2000;
    int         max_attempts      = 4;
    int         enqueue_timeout_ms = 100;
    int         drain_timeout_ms  = 10000;
    int         health_timeout_ms = 5000;
    std::size_t outbound_max_fragments = 64;
    std::size_t outbound_max_estimated_bytes = 64ULL * 1024ULL * 1024ULL;
    std::string aiserver_id       = "aiserver-0";
    std::string env_id            = "env-0";
};

struct MetricsConfig {
    std::size_t episode_window = 100;
};

// ---- AIServer 完整配置 ----
struct AIServerConfig {
    ServerConfig   server;
    StrategyConfig strategy;
    ContractConfig contract;
    TrainingSemanticsConfig training_semantics;
    PolicyConfig policy;
    ObservationConfig observation;
    MazeRewardConfig reward;
    SingleMapTaskControllerConfig curriculum;
    ModelConfig    model;
    ModelDistributionConfig model_distribution;
    MazeTaskConfig task;
    SampleOutputConfig sample_output;
    MetricsConfig metrics;
};

// ---- 配置加载器 ----
// Load and validate the complete immutable training identity. Missing or
// malformed critical identity fields fail closed.
bool LoadServerConfig(const std::string& yaml_path, AIServerConfig& out_config);
