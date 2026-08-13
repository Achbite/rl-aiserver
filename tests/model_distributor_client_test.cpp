#include "model/model_distributor_client.h"
#include "model/model_manifest.h"
#include "training.grpc.pb.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <openssl/evp.h>

#include <array>
#include <atomic>
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
        "7e3eb7227e67a2a880130c9c82f87041691c0095f838a60f80abc1f387c1c5b3";
    config.contract.artifact_digest.hex =
        "077ac6d61486fafd5f0430eeb05a492764b36e073282f6d7626d0414bb5b2ddf";
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
                                             const std::string& checksum,
                                             ModelVersion version = 0) {
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
    identity->set_model_version(version);
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
    manifest.set_artifact_uri(
        "file:///models/" +
        ModelDistributorClient::CacheVersionDirectoryName(version) +
        "/SaveModel.onnx");
    manifest.set_model_file("SaveModel.onnx");
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
    manifest.set_train_updates(0);
    manifest.set_published_at_unix_ms(1);
    manifest.set_ready(true);
    auto digest_source = manifest;
    digest_source.mutable_identity()->clear_manifest_digest();
    SetDigest(Sha256(DeterministicBytes(digest_source)),
              manifest.mutable_identity()->mutable_manifest_digest());
    return manifest;
}

bool WriteCachedPublication(const AIServerConfig& config,
                            const std::filesystem::path& cache_root,
                            ModelVersion version,
                            const std::string& payload,
                            std::string& error) {
    namespace fs = std::filesystem;
    const fs::path directory =
        cache_root /
        ModelDistributorClient::CacheVersionDirectoryName(version);
    std::error_code fs_error;
    fs::create_directories(directory, fs_error);
    if (fs_error) {
        error = "cannot create cached publication fixture";
        return false;
    }
    const fs::path model = directory / "SaveModel.onnx";
    {
        std::ofstream output(model, std::ios::binary | std::ios::trunc);
        output << payload;
        if (!output) {
            error = "cannot write cached publication fixture";
            return false;
        }
    }
    std::string checksum;
    if (!ComputeFileSha256(model.string(), checksum, error)) return false;
    auto manifest = MakeManifest(config, checksum, version);
    manifest.set_size_bytes(static_cast<int64_t>(payload.size()));
    auto digest_source = manifest;
    digest_source.mutable_identity()->clear_manifest_digest();
    SetDigest(Sha256(DeterministicBytes(digest_source)),
              manifest.mutable_identity()->mutable_manifest_digest());
    return WriteModelManifestFile(
        manifest, (directory / config.model.manifest_name).string(), error);
}

class CorruptModelService final
    : public training::ModelDistributorService::Service {
public:
    explicit CorruptModelService(training::ModelArtifactManifest manifest)
        : manifest_(std::move(manifest)) {}

    grpc::Status GetModelManifest(
        grpc::ServerContext*, const training::GetModelManifestReq* request,
        training::GetModelManifestRsp* response) override {
        last_manifest_requester_.CopyFrom(request->requester());
        response->set_ret_code(0);
        response->mutable_manifest()->CopyFrom(manifest_);
        if (!omit_range_.load()) {
            response->set_available_floor_model_version(
                manifest_.identity().model_version());
            response->set_latest_available_model_version(
                manifest_.identity().model_version());
        }
        return grpc::Status::OK;
    }

    grpc::Status DownloadModel(
        grpc::ServerContext*, const training::DownloadModelReq* request,
        grpc::ServerWriter<training::ModelChunk>* writer) override {
        last_download_requester_.CopyFrom(request->requester());
        training::ModelChunk chunk;
        chunk.mutable_model()->CopyFrom(request->requested_model());
        chunk.set_offset(0);
        chunk.set_data(corrupt_.load() ? "evil" : "good");
        writer->Write(chunk);
        return grpc::Status::OK;
    }

    void SetCorrupt(bool corrupt) { corrupt_.store(corrupt); }
    void SetOmitRange(bool omit) { omit_range_.store(omit); }
    void SetManifest(training::ModelArtifactManifest manifest) {
        manifest_ = std::move(manifest);
    }
    const common::ServiceInstanceIdentity& last_manifest_requester() const {
        return last_manifest_requester_;
    }
    const common::ServiceInstanceIdentity& last_download_requester() const {
        return last_download_requester_;
    }

private:
    training::ModelArtifactManifest manifest_;
    std::atomic<bool> corrupt_{true};
    std::atomic<bool> omit_range_{false};
    common::ServiceInstanceIdentity last_manifest_requester_;
    common::ServiceInstanceIdentity last_download_requester_;
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
    ModelDistributorClient client(config, "producer-instance-7", 7);
    ModelManifest manifest;
    ModelVersion latest_version = 0;
    std::string latest_checksum;
    if (!client.GetLatestIdentity("aiserver-0", latest_version,
                                  latest_checksum, error) ||
        latest_version != 0 || latest_checksum != checksum) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("latest model identity query failed: " + error);
    }
    if (service.last_manifest_requester().component() != "rl-aiserver" ||
        service.last_manifest_requester().instance_id() !=
            "producer-instance-7" ||
        service.last_manifest_requester().lifecycle_epoch() != 7) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("manifest request did not use process producer identity");
    }
    ModelDistributorClient::AvailableRange range;
    if (!client.GetAvailableRange("aiserver-0", range, error) ||
        range.floor_model_version != 0 ||
        range.latest_model_version != 0 ||
        range.latest_checksum != checksum) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("available model range query failed: " + error);
    }
    service.SetOmitRange(true);
    error.clear();
    if (client.GetAvailableRange("aiserver-0", range, error) ||
        error != "Model Distributor available range is missing") {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("missing available range did not fail closed: " + error);
    }
    service.SetOmitRange(false);
    if (client.FetchLatest("aiserver-0", manifest, error) ||
        error != "downloaded model checksum mismatch") {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("corrupted transfer was not rejected: " + error);
    }
    const fs::path cache_root = root / "local-train" / "cache";
    if (fs::exists(cache_root) &&
        fs::directory_iterator(cache_root) != fs::directory_iterator{}) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("corrupted transfer left a partial cache directory");
    }
    service.SetCorrupt(false);
    error.clear();
    if (!client.FetchVersion("aiserver-0", 0, manifest, error) ||
        fs::path(manifest.model_path).parent_path().filename().string().rfind(
            ".tmp-000000-", 0) != 0 ||
        !client.PublishPrepared(manifest, error) ||
        manifest.model_path !=
            (cache_root / "000000" / "SaveModel.onnx").string() ||
        !fs::is_regular_file(manifest.model_path) ||
        !fs::is_regular_file(cache_root / "000000" / "manifest.json")) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("complete model was not atomically published: " + error);
    }
    if (service.last_download_requester().component() != "rl-aiserver" ||
        service.last_download_requester().instance_id() !=
            "producer-instance-7" ||
        service.last_download_requester().lifecycle_epoch() != 7) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("download request did not use process producer identity");
    }
    const ModelVersion high_version =
        static_cast<ModelVersion>(std::numeric_limits<int>::max()) + 1;
    service.SetManifest(MakeManifest(config, checksum, high_version));
    error.clear();
    if (!client.FetchLatest("aiserver-0", manifest, error) ||
        manifest.model_version != high_version ||
        fs::path(manifest.model_path).parent_path().filename().string().rfind(
            ".tmp-2147483648-", 0) != 0 ||
        !client.DiscardTemporary(manifest, error)) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("latest fetch preserves a version above INT_MAX: " + error);
    }
    service.SetManifest(MakeManifest(config, checksum, 0));
    std::vector<ModelManifest> recovered;
    if (!client.RecoverCache(recovered, error) || recovered.size() != 1 ||
        recovered.front().model_version != 0) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("published model was not recovered: " + error);
    }

    const fs::path abandoned_download =
        cache_root /
        (".tmp-000001-" + std::to_string(::getpid()) + "-1");
    const fs::path abandoned_prune =
        cache_root /
        (".prune-000002-" + std::to_string(::getpid()) + "-2");
    fs::create_directories(abandoned_download);
    fs::create_directories(abandoned_prune);
    std::ofstream(abandoned_download / "partial") << "partial";
    std::ofstream(abandoned_prune / "partial") << "partial";
    if (!client.RecoverCache(recovered, error) ||
        fs::exists(abandoned_download) || fs::exists(abandoned_prune)) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("private cache crash residue was not recovered: " + error);
    }

    const fs::path corrupt_version = cache_root / "000001";
    fs::create_directories(corrupt_version);
    std::ofstream(corrupt_version / "SaveModel.onnx") << "bad";
    if (!client.RecoverCache(recovered, error) ||
        fs::exists(corrupt_version)) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("corrupt canonical cache version was not removed: " +
                    error);
    }

    const fs::path unknown_entry = cache_root / ".unknown-cache-entry";
    fs::create_directories(unknown_entry);
    error.clear();
    if (client.RecoverCache(recovered, error) ||
        error.find("unrecognized entry") == std::string::npos ||
        !fs::exists(unknown_entry)) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("unknown cache entry did not fail closed: " + error);
    }
    fs::remove_all(unknown_entry);

    for (int version = 1; version <= 101; ++version) {
        if (!WriteCachedPublication(
                config, cache_root, version, "good", error)) {
            server->Shutdown();
            fs::remove_all(root);
            return Fail("cannot create retention fixture: " + error);
        }
    }
    if (!client.RecoverCache(recovered, error) ||
        recovered.size() !=
            ModelDistributorClient::kCacheRetentionVersions ||
        recovered.front().model_version != 1 ||
        recovered.back().model_version != 101 ||
        fs::exists(cache_root / "000000")) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("102-version cache did not converge to versions 1..101: " +
                    error);
    }
    std::optional<ModelVersion> missing_version;
    if (!client.GetFirstMissingCachedVersion(
            1, 101, missing_version, error) || missing_version.has_value()) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("complete cache window reported a false gap: " + error);
    }
    fs::remove_all(cache_root / "000050");
    if (!client.RecoverCache(recovered, error) ||
        !client.GetFirstMissingCachedVersion(
            1, 101, missing_version, error) ||
        missing_version != std::optional<ModelVersion>(50)) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("cache gap was not reported exactly: " + error);
    }
    server->Shutdown();

    config.model_distribution.port = 1;
    config.model_distribution.rpc_timeout_ms = 100;
    ModelDistributorClient unavailable(config, "producer-instance-8", 8);
    error.clear();
    if (unavailable.FetchLatest("aiserver-0", manifest, error) ||
        error.find("model manifest RPC failed") == std::string::npos) {
        fs::remove_all(root);
        return Fail("unreachable ModelDistributor was not reported");
    }

    if (ModelDistributorClient::CacheVersionDirectoryName(0) != "000000" ||
        ModelDistributorClient::CacheVersionDirectoryName(1000000) !=
            "1000000" ||
        ModelDistributorClient::CacheVersionDirectoryName(
            std::numeric_limits<ModelVersion>::max()) !=
            "18446744073709551615") {
        fs::remove_all(root);
        return Fail("cache directory names are not minimum-width six");
    }

    std::optional<ModelVersion> high_missing;
    if (!unavailable.GetFirstMissingCachedVersion(
            std::numeric_limits<ModelVersion>::max() - 1,
            std::numeric_limits<ModelVersion>::max(), high_missing, error) ||
        high_missing != std::optional<ModelVersion>(
                            std::numeric_limits<ModelVersion>::max() - 1)) {
        fs::remove_all(root);
        return Fail("uint64 cache gap scan did not terminate safely: " + error);
    }

    fs::remove_all(root);
    std::cout << "model_distributor_client_contract: PASS\n";
    return 0;
}
