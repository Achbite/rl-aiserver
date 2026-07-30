#pragma once

#include "config/config_loader.h"

#include <cstdint>
#include <string>
#include <vector>

struct ModelManifest {
    int schema_version = 0;
    std::string contract_version;
    std::string run_id;
    int model_version = -1;
    std::string artifact_uri;
    std::string model_file;
    int64_t size_bytes = 0;
    std::string sha256;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> action_shape;
    std::vector<int64_t> value_shape;
    int64_t seed = 0;
    int64_t published_ts_ms = 0;
    bool ready = false;
    std::string model_path;
    std::string manifest_path;
};

bool LoadModelManifest(const ModelConfig& config,
                       const std::string& run_id,
                       ModelManifest& manifest,
                       std::string& error);

bool LoadModelManifestFile(const ModelConfig& config,
                           const std::string& manifest_path,
                           const std::string& expected_run_id,
                           ModelManifest& manifest,
                           std::string& error);

bool ComputeFileSha256(const std::string& path,
                       std::string& checksum,
                       std::string& error);
