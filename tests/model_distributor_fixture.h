#pragma once
#include "training.grpc.pb.h"
#include "contracts/training_namespaces.h"
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace model_fixture {
std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) throw std::runtime_error("open fixed ONNX fixture");
    std::ostringstream output;
    output << input.rdbuf();
    if (input.bad()) throw std::runtime_error("read fixed ONNX fixture");
    return output.str();
}

training::ModelArtifactManifest MakeManifest(
    const std::string& model_bytes) {
    training::ModelArtifactManifest manifest;
    auto* identity = manifest.mutable_identity();
    identity->set_model_lineage_id("lineage-fixed");
    identity->set_model_step(0);
    manifest.set_size_bytes(static_cast<int64_t>(model_bytes.size()));
    manifest.set_trained_samples(0);
    manifest.set_published_at_unix_ms(1700000000000);
    return manifest;
}

class FixedModelDistributor final
    : public training::ModelDistributorService::Service {
public:
    FixedModelDistributor(training::ModelArtifactManifest manifest,
                          std::string model_bytes)
        : manifest_(std::move(manifest)),
          model_bytes_(std::move(model_bytes)) {}

    grpc::Status GetModelDistributorStatus(
        grpc::ServerContext*,
        const training::ModelDistributorStatusReq*,
        training::ModelDistributorStatusRsp* response) override {
        std::lock_guard<std::mutex> lock(mutex_);
        response->set_ready(true);
        FillAuthority(response->mutable_distributor());
        *response->mutable_latest_model() = manifest_.identity();
        response->set_available_floor_model_step(0);
        response->set_latest_available_model_step(manifest_.identity().model_step());
        return grpc::Status::OK;
    }

    grpc::Status GetModelManifest(
        grpc::ServerContext*,
        const training::GetModelManifestReq* request,
        training::GetModelManifestRsp* response) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (request->requested_model().model_lineage_id() !=
                manifest_.identity().model_lineage_id() ||
            !request->requested_model().has_model_step() ||
            request->requested_model().model_step() != manifest_.identity().model_step()) {
            response->set_result(training::MODEL_LOOKUP_RESULT_NOT_FOUND);
            response->set_message("fixed model not found");
            return grpc::Status::OK;
        }
        response->set_result(training::MODEL_LOOKUP_RESULT_FOUND);
        *response->mutable_manifest() = manifest_;
        FillAuthority(response->mutable_distributor());
        response->set_available_floor_model_step(0);
        response->set_latest_available_model_step(manifest_.identity().model_step());
        return grpc::Status::OK;
    }

    grpc::Status DownloadModel(
        grpc::ServerContext*,
        const training::DownloadModelReq* request,
        grpc::ServerWriter<training::ModelChunk>* writer) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (request->requested_model().SerializeAsString() !=
            manifest_.identity().SerializeAsString()) {
            return grpc::Status(
                grpc::StatusCode::NOT_FOUND, "fixed model not found");
        }
        training::ModelChunk chunk;
        *chunk.mutable_model() = manifest_.identity();
        chunk.set_offset(0);
        chunk.set_data(model_bytes_);
        writer->Write(chunk);
        return grpc::Status::OK;
    }

    grpc::Status AckModel(grpc::ServerContext*,
                          const training::AckModelReq* request,
                          training::AckModelRsp* response) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ack_ = *request;
        response->set_result(training::MODEL_ACK_RESULT_APPLIED);
        FillAuthority(response->mutable_distributor());
        return grpc::Status::OK;
    }

    void SetCandidate(uint64_t step, std::string bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        model_bytes_ = std::move(bytes);
        manifest_.mutable_identity()->set_model_step(step);
        manifest_.set_size_bytes(static_cast<int64_t>(model_bytes_.size()));
    }

    training::AckModelReq ack() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ack_;
    }

private:
    static void FillAuthority(common::ServiceInstanceIdentity* identity) {
        identity->set_component("model-distributor");
        identity->set_instance_id("model-distributor-fixed");
        identity->set_lifecycle_epoch(1);
    }

    training::ModelArtifactManifest manifest_;
    std::string model_bytes_;
    mutable std::mutex mutex_;
    training::AckModelReq ack_;
};

class TemporaryRoot {
public:
    TemporaryRoot()
        : path_(std::filesystem::temp_directory_path() /
                ("aiserver-model-update-" + std::to_string(::getpid()))) {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~TemporaryRoot() { std::filesystem::remove_all(path_); }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};


}  // namespace model_fixture
