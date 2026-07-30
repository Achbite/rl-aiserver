#pragma once

#include <cstddef>
#include <string>

#include "config/run_mode.h"

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
    std::string local_dir     = "models/local";   // 本地模型目录（推理优先）
    std::string p2p_dir       = "models/cache";
    std::string smoke_dir     = "models/smoke";
    std::string save_name     = "SaveModel";      // 本地保存的模型文件名（不含扩展名）
    std::string manifest_name = "manifest.json";
    int         startup_timeout_ms = 30000;
    int         expected_obs_dim = 13;
    int         expected_action_dim = 9;
};

struct ModelDistributionConfig {
    std::string host = "maze-learner";
    int port = 9200;
    int poll_interval_ms = 200;
    int boundary_wait_ms = 1000;
    int rpc_timeout_ms = 5000;
    std::string contract_version = "0.3.0";
};

// ---- 样本分发服务连接参数 ----
struct SampleOutputConfig {
    bool        enabled           = true;                   // 是否向 SampleDistributor 推送样本
    std::string host              = "127.0.0.1";
    int         port              = 9100;
    int         fragment_samples  = 128;
    int         rpc_timeout_ms    = 2000;
    int         max_attempts      = 4;
    int         enqueue_timeout_ms = 100;
    int         drain_timeout_ms  = 10000;
    int         health_timeout_ms = 5000;
    std::size_t outbound_max_fragments = 64;
    std::size_t outbound_max_estimated_bytes = 64ULL * 1024ULL * 1024ULL;
    std::string run_id            = "local-run";
    std::string aiserver_id       = "aiserver-0";
    std::string env_id            = "env-0";
};

// ---- AIServer 完整配置 ----
struct AIServerConfig {
    ServerConfig   server;
    StrategyConfig strategy;
    ModelConfig    model;
    ModelDistributionConfig model_distribution;
    SampleOutputConfig sample_output;
};

// ---- 配置加载器 ----
// 从 YAML 文件加载配置，失败时使用默认值
bool LoadServerConfig(const std::string& yaml_path, AIServerConfig& out_config);
