#include "ai/onnx_inferencer.h"
#include "model/model_distributor_client.h"
#include "model/model_manifest.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <grpcpp/grpcpp.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <openssl/evp.h>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
}

void SetDigest(const std::string& hex, common::ContentDigest* digest) {
    digest->set_algorithm(common::DIGEST_ALGORITHM_SHA256);
    digest->set_hex(hex);
}

std::string Sha256(const std::string& data) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    Require(context != nullptr, "create SHA-256 context");
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_size = 0;
    const bool ok =
        EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
        EVP_DigestUpdate(context, data.data(), data.size()) == 1 &&
        EVP_DigestFinal_ex(context, digest, &digest_size) == 1;
    EVP_MD_CTX_free(context);
    Require(ok && digest_size == 32, "compute SHA-256");
    static constexpr char kHex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (unsigned int index = 0; index < digest_size; ++index) {
        result.push_back(kHex[digest[index] >> 4]);
        result.push_back(kHex[digest[index] & 0x0f]);
    }
    return result;
}

template <typename Message>
std::string DeterministicBytes(const Message& message) {
    std::string serialized;
    google::protobuf::io::StringOutputStream output(&serialized);
    google::protobuf::io::CodedOutputStream coded(&output);
    coded.SetSerializationDeterministic(true);
    Require(message.SerializeToCodedStream(&coded) && !coded.HadError(),
            "serialize protobuf deterministically");
    coded.Trim();
    return serialized;
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    Require(input.is_open(), "open fixed ONNX fixture");
    std::ostringstream output;
    output << input.rdbuf();
    Require(!input.bad(), "read fixed ONNX fixture");
    return output.str();
}

void FillSchema(const SchemaConfig& source,
                common::SchemaIdentity* destination) {
    destination->set_schema_id(source.schema_id);
    destination->set_schema_version(source.schema_version);
    SetDigest(source.canonical_digest.hex,
              destination->mutable_canonical_digest());
}

void FillContract(const ContractConfig& source,
                  common::ContractIdentity* destination) {
    destination->set_package_name(source.package_name);
    destination->set_package_version(source.package_version);
    SetDigest(source.source_digest.hex, destination->mutable_source_digest());
    SetDigest(source.artifact_digest.hex,
              destination->mutable_artifact_digest());
    destination->set_platform(source.platform);
    destination->set_generator_identity(source.generator_identity);
}

AIServerConfig MakeConfig(const std::filesystem::path& root, int port) {
    AIServerConfig config;
    config.contract.source_digest.hex = std::string(64, '1');
    config.contract.artifact_digest.hex = std::string(64, '2');
    config.contract.generator_identity = std::string(64, '3');
    config.training_semantics.observation_schema = {
        "maze.observation.v3", 1, {"sha256", std::string(64, '4')}};
    config.training_semantics.action_schema = {
        "maze.action.v1", 1, {"sha256", std::string(64, '5')}};
    config.training_semantics.reward_schema = {
        "maze.reward.v4", 1, {"sha256", std::string(64, '6')}};
    config.training_semantics.semantics_digest.hex = std::string(64, '7');
    config.model.local_train_dir = (root / "train").string();
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    config.model_distribution.rpc_timeout_ms = 1000;
    return config;
}

training::ModelArtifactManifest MakeManifest(
    const AIServerConfig& config,
    const std::string& model_bytes) {
    training::ModelArtifactManifest manifest;
    manifest.set_manifest_schema_version(3);
    FillContract(config.contract, manifest.mutable_contract());
    auto* identity = manifest.mutable_identity();
    identity->set_model_lineage_id("lineage-fixed");
    identity->set_model_step(0);
    SetDigest(Sha256(model_bytes), identity->mutable_artifact_digest());
    FillSchema(config.training_semantics.observation_schema,
               manifest.mutable_observation_schema());
    FillSchema(config.training_semantics.action_schema,
               manifest.mutable_action_schema());
    manifest.set_model_architecture_id(config.model.model_architecture_id);
    manifest.set_tensor_dtype(config.model.tensor_dtype);
    manifest.add_input_shape(1);
    manifest.add_input_shape(config.model.expected_obs_dim);
    manifest.add_action_shape(1);
    manifest.add_action_shape(config.model.expected_action_dim);
    manifest.add_value_shape(1);
    manifest.add_value_shape(1);
    manifest.set_artifact_uri("file:///models/0000000/SaveModel.onnx");
    manifest.set_model_file("SaveModel.onnx");
    manifest.set_size_bytes(static_cast<int64_t>(model_bytes.size()));
    manifest.set_seed(0);
    manifest.set_train_updates(0);
    manifest.set_trained_samples(0);
    SetDigest(std::string(64, '8'),
              manifest.mutable_training_config_digest());

    auto* semantics = manifest.mutable_training_semantics();
    semantics->set_training_contract_id(
        config.training_semantics.training_contract_id);
    FillSchema(config.training_semantics.observation_schema,
               semantics->mutable_observation_schema());
    FillSchema(config.training_semantics.action_schema,
               semantics->mutable_action_schema());
    FillSchema(config.training_semantics.reward_schema,
               semantics->mutable_reward_schema());
    semantics->set_policy_distribution_schema_id(
        config.training_semantics.policy_distribution_schema_id);
    semantics->set_model_architecture_id(
        config.training_semantics.model_architecture_id);
    SetDigest(config.training_semantics.semantics_digest.hex,
              semantics->mutable_semantics_digest());

    auto* profile = manifest.mutable_rollout_estimator_profile();
    profile->set_profile_schema_version(1);
    profile->set_gamma(0.99);
    profile->set_gae_lambda(0.95);
    profile->set_tmax(128);
    profile->set_gae_formula_id("gae.backward.v1");
    profile->set_terminal_bootstrap_semantics_id(
        "maze.timeout-keep-and-cut-bootstrap.v1");
    profile->set_value_target_formula_id(
        "advantage-plus-behavior-value.v1");
    profile->set_value_head_abi_id("scalar-value.float32.v1");
    SetDigest(config.training_semantics.reward_schema.canonical_digest.hex,
              profile->mutable_reward_semantics_digest());
    profile->set_numeric_dtype("float32");
    profile->set_finite_rule_id("reject-nonfinite.v1");
    profile->set_model_pin_semantics_id("per-agent-segment-pin.v1");
    profile->clear_profile_digest();
    const std::string profile_digest = Sha256(DeterministicBytes(*profile));
    SetDigest(profile_digest, profile->mutable_profile_digest());

    manifest.set_published_at_unix_ms(1700000000000);
    manifest.set_ready(true);
    manifest.mutable_identity()->clear_manifest_digest();
    const std::string manifest_digest = Sha256(DeterministicBytes(manifest));
    SetDigest(manifest_digest,
              manifest.mutable_identity()->mutable_manifest_digest());
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
        response->set_ready(true);
        *response->mutable_contract() = manifest_.contract();
        FillAuthority(response->mutable_distributor());
        *response->mutable_latest_model() = manifest_.identity();
        response->set_available_floor_model_step(0);
        response->set_latest_available_model_step(0);
        return grpc::Status::OK;
    }

    grpc::Status GetModelManifest(
        grpc::ServerContext*,
        const training::GetModelManifestReq* request,
        training::GetModelManifestRsp* response) override {
        if (request->requested_model().model_lineage_id() !=
                manifest_.identity().model_lineage_id() ||
            !request->requested_model().has_model_step() ||
            request->requested_model().model_step() != 0) {
            response->set_ret_code(1);
            response->set_message("fixed model not found");
            return grpc::Status::OK;
        }
        response->set_ret_code(0);
        *response->mutable_manifest() = manifest_;
        FillAuthority(response->mutable_distributor());
        response->set_available_floor_model_step(0);
        response->set_latest_available_model_step(0);
        return grpc::Status::OK;
    }

    grpc::Status DownloadModel(
        grpc::ServerContext*,
        const training::DownloadModelReq* request,
        grpc::ServerWriter<training::ModelChunk>* writer) override {
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
        response->set_ret_code(0);
        response->set_result(training::MODEL_ACK_RESULT_APPLIED);
        FillAuthority(response->mutable_distributor());
        return grpc::Status::OK;
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

void TestModelUpdate(const std::string& fixture_path) {
    TemporaryRoot root;
    const std::string model_bytes = ReadFile(fixture_path);
    AIServerConfig config = MakeConfig(root.path(), 0);
    const auto wire_manifest = MakeManifest(config, model_bytes);
    FixedModelDistributor distributor(wire_manifest, model_bytes);

    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(
        "127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&distributor);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "start local ModelDistributor test service");
    config.model_distribution.port = port;

    ModelDistributorClient client(config, "aiserver-fixed", 1);
    std::string error;
    ModelDistributorClient::AvailableRange range;
    Require(client.GetAvailableRange("aiserver-fixed", range, error) &&
                range.floor_model_step == 0 &&
                range.latest_model_step == 0,
            "discover the fixed model identity: " + error);
    std::vector<ModelManifest> recovered_models;
    ModelDistributorClient::CacheRecoveryFacts cache_recovery;
    Require(client.RecoverCache(
                recovered_models, cache_recovery, error),
            "initialise the production model cache: " + error);

    ModelManifest downloaded;
    Require(client.FetchStep("aiserver-fixed", 0, downloaded, error) &&
                downloaded.wire.identity().SerializeAsString() ==
                    wire_manifest.identity().SerializeAsString() &&
                ReadFile(downloaded.model_path) == model_bytes,
            "download and verify the fixed model: " + error);

    OnnxInferencer inferencer;
    OnnxInferencer::PreparedModel prepared;
    Require(inferencer.PrepareModel(
                downloaded.model_path, config.model.expected_obs_dim,
                config.model.expected_action_dim, prepared, &error) &&
                prepared.valid(),
            "Prepare the downloaded ONNX model: " + error);
    Require(client.PublishPrepared(downloaded, error),
            "publish the prepared model into the private cache: " + error);
    prepared.model_path = downloaded.model_path;
    Require(client.Ack(downloaded, "aiserver-fixed",
                       training::MODEL_LOAD_STATUS_LOADED, "loaded", error),
            "ACK the exact prepared model: " + error);
    inferencer.ActivatePreparedModel(std::move(prepared));

    std::vector<float> logits;
    float value = 0.0f;
    Require(inferencer.Infer(
                std::vector<float>(config.model.expected_obs_dim, 0.25f),
                config.model.expected_obs_dim, logits, value) &&
                logits.size() ==
                    static_cast<std::size_t>(config.model.expected_action_dim) &&
                std::isfinite(value),
            "the activated model produces finite logits and value");
    for (float logit : logits) {
        Require(std::isfinite(logit), "the activated model logit is finite");
    }

    const auto ack = distributor.ack();
    Require(ack.model().SerializeAsString() ==
                wire_manifest.identity().SerializeAsString() &&
                ack.load_status() == training::MODEL_LOAD_STATUS_LOADED,
            "the ACK belongs to the downloaded and activated model");

    server->Shutdown();
    server->Wait();
}

}  // namespace

int main(int argc, char** argv) {
    Require(argc == 2, "usage: model_update_development_test MODEL");
    TestModelUpdate(argv[1]);
    std::cout << "aiserver_model_update_development_contract: PASS"
              << std::endl;
    return 0;
}
