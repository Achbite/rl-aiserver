#pragma once

#include "config/config_loader.h"
#include "contracts/contract_namespaces.h"
#include "training.grpc.pb.h"
#include "model/model_manifest.h"

#include <grpcpp/grpcpp.h>

#include <memory>
#include <cstddef>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

class ModelDistributorClient {
public:
    static constexpr std::size_t kCacheRetentionVersions = 101;
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
        ModelVersion floor_model_version = 0;
        ModelVersion latest_model_version = 0;
        std::string latest_checksum;
    };

    ModelDistributorClient(const AIServerConfig& config,
                           std::string producer_instance_id,
                           uint64_t producer_lifecycle_epoch);

    bool FetchLatest(const std::string& aiserver_id,
                     ModelManifest& manifest,
                     std::string& error);

    bool FetchVersion(const std::string& aiserver_id,
                      ModelVersion model_version,
                      ModelManifest& manifest,
                      std::string& error);

    bool GetLatestIdentity(const std::string& aiserver_id,
                           ModelVersion& model_version,
                           std::string& checksum,
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

    bool RecoverCache(std::vector<ModelManifest>& models,
                      std::string& error);
    bool LoadCachedVersion(ModelVersion model_version,
                           ModelManifest& manifest,
                           std::string& error) const;
    bool PublishPrepared(ModelManifest& manifest,
                         std::string& error);
    bool DiscardTemporary(const ModelManifest& manifest,
                          std::string& error) const;
    bool GetFirstMissingCachedVersion(
                                      ModelVersion floor_model_version,
                                      ModelVersion latest_model_version,
                                      std::optional<ModelVersion>& missing_model_version,
                                      std::string& error) const;
    bool PruneCache(std::string& error);

    static std::string CacheVersionDirectoryName(ModelVersion model_version);

private:
    bool ValidateManifest(const training::ModelArtifactManifest& source,
                          std::optional<ModelVersion> expected_version,
                          std::string& error) const;
    bool DownloadToTemporary(
        const training::ModelArtifactManifest& source,
        const std::string& aiserver_id,
        std::string& local_path,
        std::string& error);
    bool Fetch(const std::string& aiserver_id,
               ModelVersion model_version,
               bool latest,
               ModelManifest& manifest,
               std::string& error);
    bool ListCachedModels(std::vector<ModelManifest>& models,
                          std::string& error) const;

    AIServerConfig config_;
    common::ServiceInstanceIdentity requester_identity_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<training::ModelDistributorService::Stub> stub_;
    mutable std::mutex cache_mutex_;
    std::set<ModelVersion> cached_versions_;
};
