#pragma once

#include "config/training_runtime_config.h"
#include "contracts/training_namespaces.h"
#include "proto/training/training.grpc.pb.h"
#include "model/model_manifest.h"

#include <grpcpp/grpcpp.h>

#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>

class ModelDistributorClient {
public:
    static constexpr const char* kCachedModelFile = "SaveModel.onnx";

    enum class AckDisposition {
        Applied,
        Rejected,
        NotApplied,
        Uncertain,
    };

    enum class AuthorityProbeDisposition {
        Ready,
        Retryable,
        Rejected,
    };

    struct AvailableRange {
        ModelStep floor_model_step = 0;
        ModelStep latest_model_step = 0;
        std::string model_lineage_id;
    };

    ModelDistributorClient(const ModelDistributionConfig& config,
                           const ModelConfig& model_config,
                           std::string producer_instance_id,
                           uint64_t producer_lifecycle_epoch);

    bool FetchLatest(const std::string& aiserver_id,
                     ModelManifest& manifest,
                     std::string& error);

    bool FetchStep(const std::string& aiserver_id,
                   ModelStep model_step,
                   ModelManifest& manifest,
                   std::string& error);

    bool GetAvailableRange(const std::string& aiserver_id,
                           AvailableRange& range,
                           std::string& error);

    bool Ack(const ModelManifest& manifest,
             const std::string& aiserver_id,
             training::ModelLoadStatus status,
             const std::string& message,
             std::string& error);
    bool ProbeAckAuthority(
        common::ServiceInstanceIdentity& authority,
        std::string& error);
    AuthorityProbeDisposition ProbeAckAuthorityDisposition(
        common::ServiceInstanceIdentity& authority,
        std::string& error);
    AckDisposition AckIdempotently(
        const ModelManifest& manifest,
        const std::string& aiserver_id,
        training::ModelLoadStatus status,
        const std::string& message,
        std::string& error,
        common::ServiceInstanceIdentity* pinned_authority = nullptr);

    bool PublishPrepared(ModelManifest& manifest,
                         std::string& error);
    bool DiscardTemporary(const ModelManifest& manifest,
                          std::string& error) const;
    bool PruneCache(const std::set<ModelStep>& protected_steps,
                    std::string& error);

    static std::string CacheStepDirectoryName(ModelStep model_step);

private:
    bool ValidateManifest(const training::ModelArtifactManifest& source,
                          std::optional<ModelStep> expected_step,
                          std::string& error) const;
    bool DownloadToTemporary(
        const training::ModelArtifactManifest& source,
        const std::string& aiserver_id,
        std::string& local_path,
        std::string& error);
    bool Fetch(const std::string& aiserver_id,
               ModelStep model_step,
               bool latest,
               ModelManifest& manifest,
               std::string& error);
    bool LoadCachedStep(ModelStep model_step,
                        ModelManifest& manifest,
                        std::string& error) const;
    bool PinModelLineage(const std::string& lineage_id,
                         std::string& error);
    bool GetPinnedModelLineage(std::string& lineage_id,
                               std::string& lineage_key,
                               std::string& error) const;

    ModelDistributionConfig config_;
    ModelConfig model_config_;
    common::ServiceInstanceIdentity requester_identity_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<training::ModelDistributorService::Stub> stub_;
    mutable std::mutex lineage_mutex_;
    std::optional<std::string> pinned_model_lineage_id_;
    std::optional<std::string> pinned_model_lineage_key_;
    mutable std::mutex cache_mutex_;
    std::set<ModelStep> cached_steps_;
};
