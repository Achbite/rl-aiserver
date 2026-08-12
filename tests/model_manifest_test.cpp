#include "model/model_manifest.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <openssl/evp.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << std::endl;
        std::exit(1);
    }
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

std::string DeterministicBytes(const google::protobuf::MessageLite& message) {
    std::string output(message.ByteSizeLong(), '\0');
    google::protobuf::io::ArrayOutputStream array(
        output.data(), static_cast<int>(output.size()));
    google::protobuf::io::CodedOutputStream coded(&array);
    coded.SetSerializationDeterministic(true);
    Require(message.SerializeToCodedStream(&coded), "serialize manifest");
    output.resize(static_cast<std::size_t>(coded.ByteCount()));
    return output;
}

std::string Sha256(const std::string& payload) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    Require(context != nullptr, "allocate SHA context");
    Require(EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1,
            "initialize SHA");
    Require(EVP_DigestUpdate(context, payload.data(), payload.size()) == 1,
            "update SHA");
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int size = 0;
    Require(EVP_DigestFinal_ex(context, digest.data(), &size) == 1,
            "finalize SHA");
    EVP_MD_CTX_free(context);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < size; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    return output.str();
}

AIServerConfig Config(const std::filesystem::path& root) {
    AIServerConfig config;
    config.contract.source_digest.hex =
        "fc1bf2e3dfd804431f2528d8da53227e55ca9b58b32fc95327558d91cebb3b97";
    config.contract.artifact_digest.hex =
        "d90083d97e377230f50c820d040a5d83ce7435dc88c4f948c222c86ac4a429ae";
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
    config.model.local_train_dir = (root / "local-train").string();
    return config;
}

training::ModelArtifactManifest Wire(const AIServerConfig& config,
                                     const std::string& artifact_digest,
                                     int version,
                                     int64_t train_updates,
                                     int64_t trained_samples) {
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
    identity->set_model_version(static_cast<uint64_t>(version));
    SetDigest(artifact_digest, identity->mutable_artifact_digest());
    SetSchema(config.training_semantics.observation_schema,
              manifest.mutable_observation_schema());
    SetSchema(config.training_semantics.action_schema,
              manifest.mutable_action_schema());
    manifest.set_model_architecture_id(config.model.model_architecture_id);
    manifest.set_tensor_dtype(config.model.tensor_dtype);
    manifest.add_input_shape(1);
    manifest.add_input_shape(config.model.expected_obs_dim);
    manifest.add_action_shape(1);
    manifest.add_action_shape(config.model.expected_action_dim);
    manifest.add_value_shape(1);
    manifest.add_value_shape(1);
    manifest.set_artifact_uri("file://model_v000000.onnx");
    manifest.set_model_file("model_v000000.onnx");
    manifest.set_size_bytes(11);
    manifest.set_seed(0);
    manifest.set_train_updates(train_updates);
    manifest.set_trained_samples(trained_samples);
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
    training::ModelArtifactManifest digest_source = manifest;
    digest_source.mutable_identity()->clear_manifest_digest();
    SetDigest(Sha256(DeterministicBytes(digest_source)),
              manifest.mutable_identity()->mutable_manifest_digest());
    return manifest;
}

void WriteSchema(std::ostream& stream,
                 const common::SchemaIdentity& schema) {
    stream << "{\"schema_id\":\"" << schema.schema_id()
           << "\",\"schema_version\":" << schema.schema_version()
           << ",\"canonical_digest\":\""
           << schema.canonical_digest().hex() << "\"}";
}

void WriteManifest(const std::filesystem::path& path,
                   const training::ModelArtifactManifest& manifest) {
    std::ofstream stream(path);
    stream << "{\"manifest_schema_version\":"
           << manifest.manifest_schema_version()
           << ",\"contract\":{\"package_name\":\""
           << manifest.contract().package_name()
           << "\",\"package_version\":\""
           << manifest.contract().package_version()
           << "\",\"source_digest\":\""
           << manifest.contract().source_digest().hex()
           << "\",\"artifact_digest\":\""
           << manifest.contract().artifact_digest().hex()
           << "\",\"platform\":\"" << manifest.contract().platform()
           << "\",\"generator_identity\":\""
           << manifest.contract().generator_identity()
           << "\"},\"identity\":{\"model_lineage_id\":\""
           << manifest.identity().model_lineage_id()
           << "\",\"model_version\":" << manifest.identity().model_version()
           << ",\"artifact_digest\":\""
           << manifest.identity().artifact_digest().hex()
           << "\",\"manifest_digest\":\""
           << manifest.identity().manifest_digest().hex() << "\"},"
           << "\"observation_schema\":";
    WriteSchema(stream, manifest.observation_schema());
    stream << ",\"action_schema\":";
    WriteSchema(stream, manifest.action_schema());
    stream << ",\"model_architecture_id\":\""
           << manifest.model_architecture_id()
           << "\",\"tensor_dtype\":\"" << manifest.tensor_dtype()
           << "\",\"input_shape\":[1," << manifest.input_shape(1)
           << "],\"action_shape\":[1," << manifest.action_shape(1)
           << "],\"value_shape\":[1,1],\"artifact_uri\":\""
           << manifest.artifact_uri() << "\",\"model_file\":\""
           << manifest.model_file() << "\",\"size_bytes\":"
           << manifest.size_bytes() << ",\"seed\":" << manifest.seed()
           << ",\"train_updates\":" << manifest.train_updates()
           << ",\"trained_samples\":" << manifest.trained_samples()
           << ",\"training_config_digest\":\""
           << manifest.training_config_digest().hex()
           << "\",\"training_semantics\":{\"training_contract_id\":\""
           << manifest.training_semantics().training_contract_id()
           << "\",\"observation_schema\":";
    WriteSchema(stream, manifest.training_semantics().observation_schema());
    stream << ",\"action_schema\":";
    WriteSchema(stream, manifest.training_semantics().action_schema());
    stream << ",\"reward_schema\":";
    WriteSchema(stream, manifest.training_semantics().reward_schema());
    stream << ",\"policy_distribution_schema_id\":\""
           << manifest.training_semantics().policy_distribution_schema_id()
           << "\",\"model_architecture_id\":\""
           << manifest.training_semantics().model_architecture_id()
           << "\",\"semantics_digest\":\""
           << manifest.training_semantics().semantics_digest().hex()
           << "\"},\"published_at_unix_ms\":"
           << manifest.published_at_unix_ms()
           << ",\"ready\":" << (manifest.ready() ? "true" : "false")
           << "}\n";
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
                          ("maze-manifest-test-" +
                           std::to_string(std::rand()));
    const fs::path active_dir = root / "local-train" / "active";
    fs::create_directories(active_dir);
    const fs::path model_path = active_dir / "model_v000000.onnx";
    {
        std::ofstream model(model_path, std::ios::binary);
        model << "model-bytes";
    }
    std::string checksum;
    std::string error;
    Require(ComputeFileSha256(model_path.string(), checksum, error),
            "calculate artifact digest");
    const AIServerConfig config = Config(root);

    auto valid = Wire(config, checksum, 0, 0, 0);
    WriteManifest(active_dir / "manifest.json", valid);
    ModelManifest loaded;
    Require(LoadModelManifest(config, loaded, error),
            "load valid 0.10.0 manifest: " + error);
    Require(loaded.model_version == 0 && loaded.sha256 == checksum &&
                loaded.observation_schema_id == "maze.observation.v3",
            "preserve model and schema identity");

    auto trained = Wire(config, checksum, 6, 6, 3072);
    WriteManifest(active_dir / "manifest.json", trained);
    Require(LoadModelManifestFile(
                config, (active_dir / "manifest.json").string(),
                loaded, error),
            "load trained model identity: " + error);
    Require(loaded.model_version == 6 && loaded.train_updates == 6 &&
                loaded.trained_samples == 3072,
            "preserve trained counters");

    auto independent_counters = Wire(config, checksum, 7, 42, 3072);
    error.clear();
    Require(ValidateModelManifest(
                config, independent_counters, -1, error),
            "publication version and train-update count are independent: " +
                error);

    auto overflowing_version = independent_counters;
    overflowing_version.mutable_identity()->set_model_version(
        static_cast<uint64_t>(std::numeric_limits<int>::max()) + 1U);
    auto overflow_digest_source = overflowing_version;
    overflow_digest_source.mutable_identity()->clear_manifest_digest();
    SetDigest(Sha256(DeterministicBytes(overflow_digest_source)),
              overflowing_version.mutable_identity()
                  ->mutable_manifest_digest());
    error.clear();
    Require(!ValidateModelManifest(
                config, overflowing_version, -1, error),
            "wire publication versions that overflow the local identity fail closed");

    auto legacy = trained;
    legacy.mutable_contract()->set_package_version("0.7.0");
    WriteManifest(active_dir / "manifest.json", legacy);
    Require(!LoadModelManifest(config, loaded, error),
            "legacy contract must fail closed");

    auto wrong_shape = trained;
    wrong_shape.set_input_shape(1, 13);
    WriteManifest(active_dir / "manifest.json", wrong_shape);
    Require(!LoadModelManifest(config, loaded, error),
            "legacy observation shape must fail closed");

    auto wrong_manifest_digest = trained;
    wrong_manifest_digest.mutable_identity()->mutable_manifest_digest()->
        set_hex(std::string(64, 'f'));
    WriteManifest(active_dir / "manifest.json", wrong_manifest_digest);
    Require(!LoadModelManifest(config, loaded, error),
            "manifest digest mismatch must fail closed");

    fs::remove_all(root);
    std::cout << "model_manifest_contract: PASS" << std::endl;
    return 0;
}
