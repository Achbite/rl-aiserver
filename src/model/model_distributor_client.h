#pragma once

#include "config/config_loader.h"
#include "contracts/contract_namespaces.h"
#include "training.grpc.pb.h"
#include "model/model_manifest.h"

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>

class ModelDistributorClient {
public:
    explicit ModelDistributorClient(const AIServerConfig& config);

    bool FetchLatest(const std::string& aiserver_id,
                     ModelManifest& manifest,
                     std::string& error);

    bool FetchVersion(const std::string& aiserver_id,
                      int model_version,
                      ModelManifest& manifest,
                      std::string& error);

    bool GetLatestIdentity(const std::string& aiserver_id,
                           int& model_version,
                           std::string& checksum,
                           std::string& error);

    bool Ack(const ModelManifest& manifest,
             const std::string& aiserver_id,
             training::ModelLoadStatus status,
             const std::string& message,
             std::string& error);

    bool Promote(ModelManifest& manifest,
                 std::string& previous_path,
                 std::string& error);

private:
    bool ValidateManifest(const training::ModelArtifactManifest& source,
                          int expected_version,
                          std::string& error) const;
    bool Download(const training::ModelArtifactManifest& source,
                  const std::string& aiserver_id,
                  std::string& local_path,
                  std::string& error);
    bool Fetch(const std::string& aiserver_id,
               int model_version,
               bool latest,
               ModelManifest& manifest,
               std::string& error);

    AIServerConfig config_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<training::ModelDistributorService::Stub> stub_;
};
