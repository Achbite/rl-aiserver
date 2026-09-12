#pragma once

#include "task/config/run_mode.h"

#include <cstddef>
#include <cstdint>
#include <string>

inline constexpr char kPublishedModelFile[] = "SaveModel.onnx";
inline constexpr char kModelManifestFile[] = "manifest.pb";

struct ServerConfig {
    int listen_port = 9002;
    int max_agents = 10;
    int run_mode = aiserver_mode::kEvaluation;
};

struct ModelConfig {
    std::string evaluation_model_path =
        "../models/eval/0000000/SaveModel.onnx";
    std::string local_train_dir = "../models/train";
    // Startup discovery/load budget and each candidate's ACK recovery budget.
    int startup_timeout_ms = 30000;
    int expected_obs_dim = 0;
    int expected_action_dim = 0;
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

struct ModelDistributionConfig {
    std::string host;
    int port = 9200;
    int poll_interval_ms = 200;
    int rpc_timeout_ms = 5000;
};

struct SampleDistributorConfig {
    bool enabled = true;
    std::string host;
    int port = 9100;
    int envelope_max_transitions = 128;
    std::size_t envelope_max_bytes = 8ULL * 1024ULL * 1024ULL;
    int rpc_timeout_ms = 2000;
    int max_attempts = 4;
    int enqueue_timeout_ms = 100;
    int drain_timeout_ms = 10000;
    int health_timeout_ms = 5000;
    int status_poll_interval_ms = 200;
    // Bound ingress recovery without changing sample delivery ownership.
    int recovery_timeout_ms = 30000;
    std::size_t outbound_max_envelopes = 64;
    std::size_t outbound_max_estimated_bytes = 64ULL * 1024ULL * 1024ULL;
    std::string aiserver_id = "aiserver-0";
    std::string env_id = "env-0";
};

struct TrainingRuntimeConfig {
    int reward_metric_interval_ms = 5000;
    ServerConfig server;
    PolicyConfig policy;
    RolloutConfig rollout;
    ModelConfig model;
    ModelDistributionConfig model_distribution;
    SampleDistributorConfig sample_distributor;
};
