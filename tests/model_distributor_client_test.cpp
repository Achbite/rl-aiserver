#include "model/model_distributor_client.h"
#include "model/model_manifest.h"
#include "training.grpc.pb.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <openssl/evp.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>

namespace {

int Fail(const std::string& message) {
    std::cerr << message << std::endl;
    return 1;
}

void SetDigest(const std::string& hex, common::ContentDigest* digest) {
    digest->set_algorithm(common::DIGEST_ALGORITHM_SHA256);
    digest->set_hex(hex);
}

void SetSchema(const SchemaConfig& source,
               common::SchemaIdentity* schema) {
    schema->set_schema_id(source.schema_id);
    schema->set_schema_version(source.schema_version);
    SetDigest(source.canonical_digest.hex,
              schema->mutable_canonical_digest());
}

std::string Sha256(const std::string& payload) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context ||
        EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(context, payload.data(), payload.size()) != 1) {
        if (context) EVP_MD_CTX_free(context);
        return "";
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int size = 0;
    if (EVP_DigestFinal_ex(context, digest.data(), &size) != 1) {
        EVP_MD_CTX_free(context);
        return "";
    }
    EVP_MD_CTX_free(context);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < size; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    return output.str();
}

std::string DeterministicBytes(const google::protobuf::MessageLite& message) {
    std::string output(message.ByteSizeLong(), '\0');
    google::protobuf::io::ArrayOutputStream array(
        output.data(), static_cast<int>(output.size()));
    google::protobuf::io::CodedOutputStream coded(&array);
    coded.SetSerializationDeterministic(true);
    if (!message.SerializeToCodedStream(&coded)) return "";
    output.resize(static_cast<std::size_t>(coded.ByteCount()));
    return output;
}

AIServerConfig MakeConfig(const std::filesystem::path& root, int port) {
    AIServerConfig config;
    config.contract.source_digest.hex =
        "861575536f18342fd427661c8f21b7b98994913e1e1c998f87fce5ee1490d438";
    config.contract.artifact_digest.hex =
        "b8e8cdabf05b15b830b27edd1555904269202042756ecf0ed8158184e57ce8f6";
    config.contract.generator_identity =
        "0eb73fc2cb675bdb34bf3db9c99dae62a82f93a5e3a72db84dcf3936464729c8";
    config.training_semantics.observation_schema = {
        "maze.observation.v3", 1,
        {"sha256", "7cee41136020f3ffc8c6ae799f630d55d0588c6a99ab7f717eac3b3d08aa18b4"}};
    config.training_semantics.action_schema = {
        "maze.action.v1", 1,
        {"sha256", "ce84c564e128f98adcc48fd420ac0df5acea61774a25de8705b602464009cfd8"}};
    config.training_semantics.reward_schema = {
        "maze.reward.v4", 1,
        {"sha256", "ed284084b79413473d5053b6d3f69320d2a4639c81451ba598ca45ac8ce15929"}};
    config.training_semantics.semantics_digest.hex =
        "6cd834542f8263135b4bfd069f372ddfdb99334060d305f58b00ce56eea10b4c";
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    config.model_distribution.rpc_timeout_ms = 1000;
    config.model.local_train_dir = (root / "local-train").string();
    return config;
}

training::ModelArtifactManifest MakeManifest(const AIServerConfig& config,
                                             const std::string& checksum) {
    training::ModelArtifactManifest manifest;
    manifest.set_manifest_schema_version(1);
    auto* contract = manifest.mutable_contract();
    contract->set_package_name(config.contract.package_name);
    contract->set_package_version(config.contract.package_version);
    SetDigest(config.contract.source_digest.hex,
              contract->mutable_source_digest());
    SetDigest(config.contract.artifact_digest.hex,
              contract->mutable_artifact_digest());
    contract->set_platform(config.contract.platform);
    contract->set_generator_identity(config.contract.generator_identity);
    auto* identity = manifest.mutable_identity();
    identity->set_model_lineage_id(config.model.expected_model_lineage_id);
    identity->set_model_version(0);
    SetDigest(checksum, identity->mutable_artifact_digest());
    SetSchema(config.training_semantics.observation_schema,
              manifest.mutable_observation_schema());
    SetSchema(config.training_semantics.action_schema,
              manifest.mutable_action_schema());
    manifest.set_model_architecture_id(config.model.model_architecture_id);
    manifest.set_tensor_dtype(config.model.tensor_dtype);
    manifest.add_input_shape(1);
    manifest.add_input_shape(17);
    manifest.add_action_shape(1);
    manifest.add_action_shape(9);
    manifest.add_value_shape(1);
    manifest.add_value_shape(1);
    manifest.set_artifact_uri("file:///models/model_v000000.onnx");
    manifest.set_model_file("model_v000000.onnx");
    manifest.set_size_bytes(4);
    SetDigest(std::string(64, 'a'),
              manifest.mutable_training_config_digest());
    auto* semantics = manifest.mutable_training_semantics();
    semantics->set_training_contract_id(
        config.training_semantics.training_contract_id);
    SetSchema(config.training_semantics.observation_schema,
              semantics->mutable_observation_schema());
    SetSchema(config.training_semantics.action_schema,
              semantics->mutable_action_schema());
    SetSchema(config.training_semantics.reward_schema,
              semantics->mutable_reward_schema());
    semantics->set_policy_distribution_schema_id(
        config.training_semantics.policy_distribution_schema_id);
    semantics->set_model_architecture_id(
        config.training_semantics.model_architecture_id);
    SetDigest(config.training_semantics.semantics_digest.hex,
              semantics->mutable_semantics_digest());
    manifest.set_published_at_unix_ms(1);
    manifest.set_ready(true);
    auto digest_source = manifest;
    digest_source.mutable_identity()->clear_manifest_digest();
    SetDigest(Sha256(DeterministicBytes(digest_source)),
              manifest.mutable_identity()->mutable_manifest_digest());
    return manifest;
}

class CorruptModelService final
    : public training::ModelDistributorService::Service {
public:
    explicit CorruptModelService(training::ModelArtifactManifest manifest)
        : manifest_(std::move(manifest)) {}

    grpc::Status GetModelManifest(
        grpc::ServerContext*, const training::GetModelManifestReq*,
        training::GetModelManifestRsp* response) override {
        response->set_ret_code(0);
        response->mutable_manifest()->CopyFrom(manifest_);
        return grpc::Status::OK;
    }

    grpc::Status DownloadModel(
        grpc::ServerContext*, const training::DownloadModelReq* request,
        grpc::ServerWriter<training::ModelChunk>* writer) override {
        training::ModelChunk chunk;
        chunk.mutable_model()->CopyFrom(request->requested_model());
        chunk.set_offset(0);
        chunk.set_data("evil");
        writer->Write(chunk);
        return grpc::Status::OK;
    }

private:
    training::ModelArtifactManifest manifest_;
};

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
                          ("model-client-test-" + std::to_string(::getpid()));
    fs::create_directories(root);
    const fs::path expected_file = root / "expected.onnx";
    std::ofstream(expected_file, std::ios::binary) << "good";
    std::string checksum;
    std::string error;
    if (!ComputeFileSha256(expected_file.string(), checksum, error)) {
        fs::remove_all(root);
        return Fail(error);
    }

    AIServerConfig config = MakeConfig(root, 0);
    CorruptModelService service(MakeManifest(config, checksum));
    int selected_port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(),
                             &selected_port);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (!server || selected_port <= 0) {
        fs::remove_all(root);
        return Fail("failed to start model service");
    }
    config.model_distribution.port = selected_port;
    ModelDistributorClient client(config);
    ModelManifest manifest;
    int latest_version = -1;
    std::string latest_checksum;
    if (!client.GetLatestIdentity("aiserver-0", latest_version,
                                  latest_checksum, error) ||
        latest_version != 0 || latest_checksum != checksum) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("latest model identity query failed: " + error);
    }
    if (client.FetchLatest("aiserver-0", manifest, error) ||
        error != "downloaded model checksum mismatch") {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("corrupted transfer was not rejected: " + error);
    }
    server->Shutdown();

    config.model_distribution.port = 1;
    config.model_distribution.rpc_timeout_ms = 100;
    ModelDistributorClient unavailable(config);
    error.clear();
    if (unavailable.FetchLatest("aiserver-0", manifest, error) ||
        error.find("model manifest RPC failed") == std::string::npos) {
        fs::remove_all(root);
        return Fail("unreachable ModelDistributor was not reported");
    }

    const fs::path incoming_dir = root / "local-train" / "incoming";
    const fs::path active_dir = root / "local-train" / "active";
    fs::create_directories(incoming_dir);
    fs::create_directories(active_dir);
    std::ofstream(active_dir / "model.onnx", std::ios::binary) << "old";
    std::ofstream(incoming_dir / "model_v000001.onnx", std::ios::binary)
        << "new";
    manifest.model_path = (incoming_dir / "model_v000001.onnx").string();
    std::string previous_path;
    error.clear();
    if (!client.Promote(manifest, previous_path, error) ||
        !fs::is_regular_file(manifest.model_path) ||
        !fs::is_regular_file(previous_path)) {
        fs::remove_all(root);
        return Fail("model promotion failed: " + error);
    }

    fs::remove_all(root);
    std::cout << "model_distributor_client_contract: PASS\n";
    return 0;
}
