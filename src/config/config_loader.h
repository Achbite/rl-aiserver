#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "config/run_mode.h"
#include "task/single_map_task_controller.h"

inline constexpr char kModelArtifactFile[] = "SaveModel.onnx";
inline constexpr char kModelManifestFile[] = "manifest.json";

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
    int         expected_obs_dim = 17;
    int         expected_action_dim = 9;
    std::string observation_schema_id = "maze.observation.v3";
    std::string action_schema_id = "maze.action.v1";
    std::string model_architecture_id = "maze.mlp-17x64x64.v1";
    std::string tensor_dtype = "float32";
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
    std::string package_version = "0.14.0";
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

struct ModelDistributionConfig {
    std::string host = "maze-learner";
    int port = 9200;
    int poll_interval_ms = 200;
    int rpc_timeout_ms = 5000;
    std::string contract_version = "0.14.0";
};

// Runtime Environment assignment owned only by AIServer. It is deliberately
// excluded from MazeTaskSpec and the task configuration digest.
struct EnvironmentConfig {
    int agent_count = 4;
};

// Maze task ownership belongs to AIServer. Client configuration cannot
// override any value in this structure.
struct MazeTaskConfig {
    std::string task_contract_id = "maze.task.v3";
    uint64_t task_revision = 3;
    DigestConfig task_config_digest{
        "sha256",
        "2502369d3df20d5c02001e7481cacd6c5be32c263cb88bdb2a40db2aee4bb167"};
    std::string fixed_map_id = "maze_117436372";
    std::string fixed_map_checksum_sha256 =
        "861e0bb22a8b9a2ed689527d080c65ec2c822367e985c49753e1be9cf3ca8ae9";
    std::string action_rule_id =
        "maze.action.9-way.no-corner-cut.v1";
    int shortest_action_steps = 188;
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

struct MetricsConfig {
    std::size_t episode_window = 100;
    std::string event_schema_catalog_path;
    SchemaConfig event_schema;
};

// ---- AIServer 完整配置 ----
struct AIServerConfig {
    ServerConfig   server;
    StrategyConfig strategy;
    ContractConfig contract;
    TrainingSemanticsConfig training_semantics;
    PolicyConfig policy;
    ObservationConfig observation;
    ModelConfig    model;
    ModelDistributionConfig model_distribution;
    EnvironmentConfig environment;
    MazeTaskConfig task;
    SampleDistributorConfig sample_distributor;
    MetricsConfig metrics;
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
// Load and validate the complete immutable training identity. Missing or
// malformed critical identity fields fail closed.
bool LoadServerConfig(const std::string& yaml_path, AIServerConfig& out_config);
bool LoadServerConfig(const std::string& yaml_path,
                      const AIServerConfigOverrides& overrides,
                      AIServerConfig& out_config,
                      AIServerConfigLoadReport& report,
                      std::string& error);
