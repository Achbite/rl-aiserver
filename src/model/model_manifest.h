#pragma once

#include "config/config_loader.h"
#include "contracts/contract_namespaces.h"
#include "training.pb.h"

#include <cstdint>
#include <string>
#include <vector>

struct ModelManifest {
    training::ModelArtifactManifest wire;
    int schema_version = 0;
    std::string contract_version;
    std::string model_lineage_id;
    int model_version = -1;
    std::string manifest_digest;
    std::string artifact_uri;
    std::string model_file;
    int64_t size_bytes = 0;
    std::string sha256;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> action_shape;
    std::vector<int64_t> value_shape;
    int64_t seed = 0;
    int64_t published_ts_ms = 0;
    std::string observation_schema_id;
    std::string action_schema_id;
    std::string model_architecture_id;
    std::string tensor_dtype;
    int64_t train_updates = 0;
    int64_t trained_samples = 0;
    bool ready = false;
    std::string model_path;
    std::string manifest_path;
};

bool ValidateModelManifest(const AIServerConfig& config,
                           const training::ModelArtifactManifest& source,
                           int expected_version,
                           std::string& error);

void AssignModelManifest(const training::ModelArtifactManifest& source,
                         const std::string& model_path,
                         ModelManifest& destination);

bool LoadModelManifest(const AIServerConfig& config,
                       ModelManifest& manifest,
                       std::string& error);

bool LoadModelManifestFile(const AIServerConfig& config,
                           const std::string& manifest_path,
                           ModelManifest& manifest,
                           std::string& error);

bool ComputeFileSha256(const std::string& path,
                       std::string& checksum,
                       std::string& error);
