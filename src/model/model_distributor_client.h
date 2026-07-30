#pragma once

#include "config/config_loader.h"
#include "maze.grpc.pb.h"
#include "model/model_manifest.h"

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>

class ModelDistributorClient {
public:
    ModelDistributorClient(const ModelDistributionConfig& distribution,
                           const ModelConfig& model);

    bool FetchLatest(const std::string& run_id,
                     const std::string& aiserver_id,
                     ModelManifest& manifest,
                     std::string& error);

    bool FetchVersion(const std::string& run_id,
                      const std::string& aiserver_id,
                      int model_version,
                      ModelManifest& manifest,
                      std::string& error);

    bool GetLatestIdentity(const std::string& run_id,
                           const std::string& aiserver_id,
                           int& model_version,
                           std::string& checksum,
                           std::string& error);

    bool Ack(const ModelManifest& manifest,
             const std::string& run_id,
             const std::string& aiserver_id,
             maze::ModelLoadStatus status,
             const std::string& message,
             std::string& error);

private:
    bool ValidateManifest(const maze::ModelArtifactManifest& source,
                          int expected_version,
                          std::string& error) const;
    bool Download(const maze::ModelArtifactManifest& source,
                  const std::string& aiserver_id,
                  std::string& local_path,
                  std::string& error);
    bool Fetch(const std::string& run_id,
               const std::string& aiserver_id,
               int model_version,
               bool latest,
               ModelManifest& manifest,
               std::string& error);

    ModelDistributionConfig distribution_;
    ModelConfig model_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<maze::ModelDistributorService::Stub> stub_;
};
