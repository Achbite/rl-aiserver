#pragma once

#include "config/config_loader.h"
#include "contracts/contract_namespaces.h"
#include "model/model_step.h"
#include "training.pb.h"

#include <cstdint>
#include <optional>
#include <string>

struct ModelManifest {
    training::ModelArtifactManifest wire;
    std::string model_path;
    std::string manifest_path;

    bool HasModelIdentity() const {
        return wire.has_identity() &&
               wire.identity().has_model_step() &&
               !wire.identity().model_lineage_id().empty();
    }

    ModelStep model_step() const { return wire.identity().model_step(); }
    const std::string& model_lineage_id() const {
        return wire.identity().model_lineage_id();
    }
    const std::string& artifact_digest() const {
        return wire.identity().artifact_digest().hex();
    }
    const std::string& manifest_digest() const {
        return wire.identity().manifest_digest().hex();
    }
    uint64_t train_updates() const { return model_step(); }
    uint64_t trained_samples() const { return wire.trained_samples(); }
};

bool ValidateModelManifest(const AIServerConfig& config,
                           const training::ModelArtifactManifest& source,
                           std::optional<ModelStep> expected_step,
                           std::string& error);

void AssignModelManifest(const training::ModelArtifactManifest& source,
                         const std::string& model_path,
                         ModelManifest& destination);

bool LoadModelManifestFile(const AIServerConfig& config,
                           const std::string& manifest_path,
                           ModelManifest& manifest,
                           std::string& error);

bool WriteModelManifestFile(
    const training::ModelArtifactManifest& manifest,
    const std::string& manifest_path,
    std::string& error);

bool ComputeFileSha256(const std::string& path,
                       std::string& checksum,
                       std::string& error);
