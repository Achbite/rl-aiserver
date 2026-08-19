#include "model/model_distributor_client.h"
#include "model/model_manifest.h"
#include "training.grpc.pb.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>
#include <openssl/evp.h>

#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

constexpr char kModelLineageA[] = "model/lineage A";
constexpr char kModelLineageB[] = "model/lineage B";

int Fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
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
        output << std::setw(2)
               << static_cast<unsigned int>(digest[index]);
    }
    return output.str();
}

std::string DeterministicBytes(
    const google::protobuf::MessageLite& message) {
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
    config.contract.source_digest.hex = std::string(64, 'a');
    config.contract.artifact_digest.hex = std::string(64, 'b');
    config.contract.generator_identity = std::string(64, 'c');
    config.training_semantics.observation_schema = {
        "maze.observation.v3", 1, {"sha256", std::string(64, 'd')}};
    config.training_semantics.action_schema = {
        "maze.action.v1", 1, {"sha256", std::string(64, 'e')}};
    config.training_semantics.reward_schema = {
        "maze.reward.transport.v1", 1,
        {"sha256", std::string(64, 'f')}};
    config.training_semantics.semantics_digest.hex =
        std::string(64, '1');
    config.model_distribution.host = "127.0.0.1";
    config.model_distribution.port = port;
    config.model_distribution.rpc_timeout_ms = 1000;
    config.model.local_train_dir = (root / "local-train").string();
    return config;
}

training::ModelArtifactManifest MakeManifest(
    const AIServerConfig& config,
    const std::string& lineage_id,
    ModelStep model_step,
    const std::string& payload) {
    training::ModelArtifactManifest manifest;
    manifest.set_manifest_schema_version(2);
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
    identity->set_model_lineage_id(lineage_id);
    identity->set_model_step(model_step);
    SetDigest(Sha256(payload), identity->mutable_artifact_digest());
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
        ModelDistributorClient::CacheStepDirectoryName(model_step) +
        "/SaveModel.onnx");
    manifest.set_model_file("SaveModel.onnx");
    manifest.set_size_bytes(static_cast<int64_t>(payload.size()));
    SetDigest(std::string(64, '2'),
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
    manifest.set_train_updates(static_cast<int64_t>(model_step));
    manifest.set_trained_samples(
        static_cast<int64_t>(model_step) * 128);
    manifest.set_published_at_unix_ms(
        static_cast<int64_t>(model_step) + 1);
    manifest.set_ready(true);
    auto digest_source = manifest;
    digest_source.mutable_identity()->clear_manifest_digest();
    SetDigest(Sha256(DeterministicBytes(digest_source)),
              manifest.mutable_identity()->mutable_manifest_digest());
    return manifest;
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream payload;
    payload << input.rdbuf();
    return input.bad() ? "" : payload.str();
}

using DirectorySnapshot = std::map<std::string, std::string>;

bool SnapshotDirectory(const std::filesystem::path& root,
                       DirectorySnapshot& snapshot,
                       std::string& error) {
    namespace fs = std::filesystem;
    snapshot.clear();
    std::error_code fs_error;
    const auto root_status = fs::symlink_status(root, fs_error);
    if (fs_error || fs::is_symlink(root_status) ||
        !fs::is_directory(root_status)) {
        error = "snapshot root is not a real directory: " + root.string();
        return false;
    }
    for (fs::recursive_directory_iterator iterator(root, fs_error), end;
         !fs_error && iterator != end; iterator.increment(fs_error)) {
        const fs::path path = iterator->path();
        const auto status = fs::symlink_status(path, fs_error);
        if (fs_error || fs::is_symlink(status)) break;
        const std::string relative = fs::relative(path, root).generic_string();
        if (fs::is_directory(status)) {
            snapshot.emplace(relative + "/", "<directory>");
        } else if (fs::is_regular_file(status)) {
            snapshot.emplace(relative, ReadFile(path));
        } else {
            error = "snapshot contains an unsupported entry: " +
                    path.string();
            return false;
        }
    }
    if (fs_error) {
        error = "cannot snapshot directory: " + fs_error.message();
        return false;
    }
    error.clear();
    return true;
}

bool ValidateLineageIdentity(const std::filesystem::path& path,
                             const std::string& lineage_id,
                             const std::string& lineage_key) {
    google::protobuf::Struct document;
    const auto status = google::protobuf::util::JsonStringToMessage(
        ReadFile(path), &document);
    if (!status.ok() || document.fields_size() != 3) return false;
    const auto& fields = document.fields();
    const auto schema = fields.find("schema_version");
    const auto id = fields.find("model_lineage_id");
    const auto key = fields.find("lineage_key");
    return schema != fields.end() &&
           schema->second.kind_case() ==
               google::protobuf::Value::kNumberValue &&
           schema->second.number_value() == 1.0 &&
           id != fields.end() &&
           id->second.kind_case() ==
               google::protobuf::Value::kStringValue &&
           id->second.string_value() == lineage_id &&
           key != fields.end() &&
           key->second.kind_case() ==
               google::protobuf::Value::kStringValue &&
           key->second.string_value() == lineage_key;
}

bool WriteCachedStep(const AIServerConfig& config,
                     const std::filesystem::path& lineage_root,
                     const std::string& lineage_id,
                     ModelStep model_step,
                     const std::string& payload,
                     std::string& error) {
    namespace fs = std::filesystem;
    const fs::path step_root =
        lineage_root /
        ModelDistributorClient::CacheStepDirectoryName(model_step);
    std::error_code fs_error;
    if (!fs::create_directory(step_root, fs_error) || fs_error) {
        error = "cannot create cached step fixture: " +
                step_root.string();
        return false;
    }
    const fs::path model_path = step_root / "SaveModel.onnx";
    std::ofstream output(model_path, std::ios::binary);
    output << payload;
    output.close();
    if (!output || !WriteModelManifestFile(
            MakeManifest(config, lineage_id, model_step, payload),
            (step_root / "manifest.json").string(), error)) {
        if (error.empty()) error = "cannot write cached step fixture";
        return false;
    }
    error.clear();
    return true;
}

std::size_t CountStepDirectories(
    const std::filesystem::path& lineage_root) {
    namespace fs = std::filesystem;
    std::size_t count = 0;
    std::error_code fs_error;
    for (fs::directory_iterator iterator(lineage_root, fs_error), end;
         !fs_error && iterator != end; iterator.increment(fs_error)) {
        const auto status = fs::symlink_status(iterator->path(), fs_error);
        if (fs_error) return 0;
        if (fs::is_directory(status) &&
            iterator->path().filename().string().front() != '.') {
            ++count;
        }
    }
    return fs_error ? 0 : count;
}

class ModelService final : public training::ModelDistributorService::Service {
public:
    ModelService(training::ModelArtifactManifest manifest,
                 std::string payload)
        : manifest_(std::move(manifest)), payload_(std::move(payload)) {}

    void SetModel(training::ModelArtifactManifest manifest,
                  std::string payload) {
        std::lock_guard<std::mutex> lock(mutex_);
        manifest_ = std::move(manifest);
        payload_ = std::move(payload);
    }

    grpc::Status GetModelManifest(
        grpc::ServerContext*, const training::GetModelManifestReq* request,
        training::GetModelManifestRsp* response) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!request->has_requested_model() ||
            request->requested_model().model_lineage_id() !=
                manifest_.identity().model_lineage_id() ||
            !request->requested_model().has_model_step() ||
            request->requested_model().model_step() !=
                manifest_.identity().model_step()) {
            response->set_ret_code(1);
            response->set_message("model not found");
            return grpc::Status::OK;
        }
        response->set_ret_code(0);
        response->mutable_manifest()->CopyFrom(manifest_);
        response->set_available_floor_model_step(
            manifest_.identity().model_step());
        response->set_latest_available_model_step(
            manifest_.identity().model_step());
        return grpc::Status::OK;
    }

    grpc::Status GetModelDistributorStatus(
        grpc::ServerContext*,
        const training::ModelDistributorStatusReq*,
        training::ModelDistributorStatusRsp* response) override {
        std::lock_guard<std::mutex> lock(mutex_);
        response->set_ready(true);
        response->mutable_contract()->CopyFrom(manifest_.contract());
        FillAuthority(response->mutable_distributor());
        response->mutable_latest_model()->CopyFrom(manifest_.identity());
        response->set_available_floor_model_step(
            manifest_.identity().model_step());
        response->set_latest_available_model_step(
            manifest_.identity().model_step());
        return grpc::Status::OK;
    }

    grpc::Status DownloadModel(
        grpc::ServerContext*, const training::DownloadModelReq* request,
        grpc::ServerWriter<training::ModelChunk>* writer) override {
        std::string payload;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (request->requested_model().SerializeAsString() !=
                manifest_.identity().SerializeAsString()) {
                return grpc::Status(
                    grpc::StatusCode::NOT_FOUND, "model not found");
            }
            payload = payload_;
        }
        ++download_count_;
        training::ModelChunk chunk;
        chunk.mutable_model()->CopyFrom(request->requested_model());
        chunk.set_offset(0);
        chunk.set_data(payload);
        writer->Write(chunk);
        return grpc::Status::OK;
    }

    grpc::Status AckModel(
        grpc::ServerContext*, const training::AckModelReq* request,
        training::AckModelRsp* response) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ack_.CopyFrom(*request);
        response->set_ret_code(0);
        response->set_result(training::MODEL_ACK_RESULT_APPLIED);
        FillAuthority(response->mutable_distributor());
        return grpc::Status::OK;
    }

    training::AckModelReq ack() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ack_;
    }

    int download_count() const { return download_count_.load(); }

private:
    static void FillAuthority(common::ServiceInstanceIdentity* identity) {
        identity->set_component("model-distributor");
        identity->set_instance_id("model-flow-distributor");
        identity->set_lifecycle_epoch(1);
    }

    mutable std::mutex mutex_;
    training::ModelArtifactManifest manifest_;
    std::string payload_;
    training::AckModelReq ack_;
    std::atomic<int> download_count_{0};
};

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
                          ("model-flow-test-" +
                           std::to_string(::getpid()));
    fs::remove_all(root);
    fs::create_directories(root);
    AIServerConfig config = MakeConfig(root, 0);
    const std::string payload_a = "lineage-a-model";
    ModelService service(
        MakeManifest(config, kModelLineageA, 0, payload_a), payload_a);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort(
        "127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (!server || port <= 0) {
        fs::remove_all(root);
        return Fail("model distributor did not start");
    }
    config.model_distribution.port = port;

    const auto FailAfterServer = [&](const std::string& message) {
        server->Shutdown();
        server->Wait();
        fs::remove_all(root);
        return Fail(message);
    };

    std::string error;
    const fs::path cache_root = root / "local-train/cache";
    const fs::path legacy_step = cache_root / "0000000";
    const fs::path legacy_private =
        cache_root /
        (".tmp-0000001-" + std::to_string(::getpid()) + "-9001");
    fs::create_directories(legacy_step);
    fs::create_directories(legacy_private);
    std::ofstream(legacy_step / "legacy.bin", std::ios::binary)
        << "legacy-canonical";
    std::ofstream(legacy_private / "legacy.tmp", std::ios::binary)
        << "legacy-private";
    DirectorySnapshot legacy_step_before;
    DirectorySnapshot legacy_private_before;
    if (!SnapshotDirectory(legacy_step, legacy_step_before, error) ||
        !SnapshotDirectory(legacy_private, legacy_private_before, error)) {
        return FailAfterServer(error);
    }

    ModelDistributorClient client(config, "aiserver-instance", 1);
    ModelDistributorClient::AvailableRange range;
    ModelDistributorClient::CacheRecoveryFacts recovery_facts;
    std::vector<ModelManifest> recovered;
    ModelManifest manifest;
    const std::string lineage_key_a = Sha256(kModelLineageA);
    const fs::path namespace_a =
        cache_root / "lineages" / lineage_key_a;
    const fs::path step_a = namespace_a / "0000000";
    if (!client.GetAvailableRange("aiserver", range, error) ||
        range.floor_model_step != 0 || range.latest_model_step != 0 ||
        range.model_lineage_id != kModelLineageA ||
        range.latest_checksum != Sha256(payload_a) ||
        !client.RecoverCache(recovered, recovery_facts, error) ||
        !recovered.empty() ||
        recovery_facts.model_lineage_key != lineage_key_a ||
        recovery_facts.ignored_legacy_entries != 2 ||
        recovery_facts.recovered_steps != 0 ||
        !client.FetchStep("aiserver", 0, manifest, error) ||
        !client.PublishPrepared(manifest, error) ||
        manifest.model_path != (step_a / "SaveModel.onnx").string() ||
        manifest.model_lineage_id != kModelLineageA ||
        manifest.model_step != 0 ||
        manifest.sha256 != range.latest_checksum ||
        manifest.manifest_digest != range.latest_manifest_digest ||
        ReadFile(step_a / "SaveModel.onnx") != payload_a ||
        !fs::is_regular_file(step_a / "manifest.json") ||
        namespace_a.filename().string() != lineage_key_a ||
        namespace_a.filename().string() == kModelLineageA ||
        !ValidateLineageIdentity(
            namespace_a / "lineage.json",
            kModelLineageA, lineage_key_a) ||
        !client.Ack(manifest, "aiserver",
                    training::MODEL_LOAD_STATUS_LOADED, "loaded", error)) {
        return FailAfterServer(
            "lineage A discovery, recovery, publication, or ACK failed: " +
            error);
    }

    const training::AckModelReq ack = service.ack();
    if (service.download_count() != 1 ||
        ack.model().model_lineage_id() != kModelLineageA ||
        ack.model().model_step() != 0 ||
        ack.load_status() != training::MODEL_LOAD_STATUS_LOADED) {
        return FailAfterServer("published model ACK identity was not preserved");
    }

    ModelDistributorClient restarted_a(
        config, "aiserver-instance-restarted", 2);
    ModelDistributorClient::AvailableRange restarted_range_a;
    ModelDistributorClient::CacheRecoveryFacts restarted_facts_a;
    ModelManifest restarted_manifest_a;
    recovered.clear();
    if (!restarted_a.GetAvailableRange(
            "aiserver", restarted_range_a, error) ||
        !restarted_a.RecoverCache(
            recovered, restarted_facts_a, error) ||
        recovered.size() != 1 ||
        restarted_facts_a.recovered_steps != 1 ||
        restarted_facts_a.ignored_legacy_entries != 2 ||
        !restarted_a.LoadCachedStep(0, restarted_manifest_a, error) ||
        restarted_manifest_a.model_lineage_id !=
            restarted_range_a.model_lineage_id ||
        restarted_manifest_a.sha256 != restarted_range_a.latest_checksum ||
        restarted_manifest_a.manifest_digest !=
            restarted_range_a.latest_manifest_digest ||
        service.download_count() != 1) {
        return FailAfterServer(
            "same-lineage restart did not reuse the exact cached model: " +
            error);
    }

    DirectorySnapshot namespace_a_before;
    if (!SnapshotDirectory(namespace_a, namespace_a_before, error)) {
        return FailAfterServer(error);
    }

    const std::string payload_b = "lineage-b-model";
    service.SetModel(
        MakeManifest(config, kModelLineageB, 0, payload_b), payload_b);
    ModelDistributorClient client_b(config, "aiserver-instance-b", 1);
    ModelDistributorClient::AvailableRange range_b;
    ModelDistributorClient::CacheRecoveryFacts recovery_facts_b;
    const std::string lineage_key_b = Sha256(kModelLineageB);
    const fs::path namespace_b =
        cache_root / "lineages" / lineage_key_b;
    const fs::path step_b = namespace_b / "0000000";
    recovered.clear();
    ModelManifest manifest_b;
    if (!client_b.GetAvailableRange("aiserver", range_b, error) ||
        range_b.model_lineage_id != kModelLineageB ||
        !client_b.RecoverCache(recovered, recovery_facts_b, error) ||
        !recovered.empty() ||
        recovery_facts_b.model_lineage_key != lineage_key_b ||
        recovery_facts_b.ignored_legacy_entries != 2 ||
        !client_b.FetchStep("aiserver", 0, manifest_b, error) ||
        !client_b.PublishPrepared(manifest_b, error) ||
        manifest_b.model_path != (step_b / "SaveModel.onnx").string() ||
        ReadFile(step_b / "SaveModel.onnx") != payload_b ||
        !ValidateLineageIdentity(
            namespace_b / "lineage.json",
            kModelLineageB, lineage_key_b) ||
        !fs::is_directory(step_a) || !fs::is_directory(step_b)) {
        return FailAfterServer(
            "same-step models did not remain isolated by lineage: " + error);
    }

    DirectorySnapshot namespace_a_after_b;
    if (!SnapshotDirectory(namespace_a, namespace_a_after_b, error) ||
        namespace_a_after_b != namespace_a_before) {
        return FailAfterServer(
            "publishing lineage B changed inactive lineage A");
    }

    DirectorySnapshot step_b_before_conflict;
    if (!SnapshotDirectory(step_b, step_b_before_conflict, error)) {
        return FailAfterServer(error);
    }
    const std::string conflicting_payload = "lineage-b-conflict";
    service.SetModel(
        MakeManifest(config, kModelLineageB, 0, conflicting_payload),
        conflicting_payload);
    ModelManifest conflicting_manifest;
    std::string conflict_error;
    if (!client_b.FetchStep(
            "aiserver", 0, conflicting_manifest, conflict_error) ||
        client_b.PublishPrepared(conflicting_manifest, conflict_error)) {
        return FailAfterServer(
            "same-lineage conflicting step did not fail closed");
    }
    std::string discard_error;
    if (!client_b.DiscardTemporary(
            conflicting_manifest, discard_error)) {
        return FailAfterServer(
            "conflicting private download did not cleanly discard: " +
            discard_error);
    }
    DirectorySnapshot step_b_after_conflict;
    if (!SnapshotDirectory(step_b, step_b_after_conflict, error) ||
        step_b_after_conflict != step_b_before_conflict) {
        return FailAfterServer(
            "same-step identity conflict overwrote the published model");
    }

    for (ModelStep step = 1; step <= 102; ++step) {
        if (!WriteCachedStep(
                config, namespace_b, kModelLineageB, step,
                "lineage-b-step-" + std::to_string(step), error)) {
            return FailAfterServer(error);
        }
    }
    if (!client_b.PruneCache({0}, error) ||
        CountStepDirectories(namespace_b) != 102 ||
        !fs::is_directory(namespace_b / "0000000") ||
        fs::exists(namespace_b / "0000001")) {
        return FailAfterServer(
            "active-lineage protected retention did not preserve step 0: " +
            error);
    }
    DirectorySnapshot namespace_a_after_protected_prune;
    if (!SnapshotDirectory(
            namespace_a, namespace_a_after_protected_prune, error) ||
        namespace_a_after_protected_prune != namespace_a_before) {
        return FailAfterServer(
            "active-lineage protected prune changed inactive lineage A");
    }
    if (!client_b.PruneCache({}, error) ||
        CountStepDirectories(namespace_b) !=
            ModelDistributorClient::kCacheRetentionSteps ||
        fs::exists(namespace_b / "0000000")) {
        return FailAfterServer(
            "active-lineage retention did not converge to 101 steps: " +
            error);
    }
    DirectorySnapshot namespace_a_after_prune;
    if (!SnapshotDirectory(namespace_a, namespace_a_after_prune, error) ||
        namespace_a_after_prune != namespace_a_before) {
        return FailAfterServer(
            "active-lineage prune changed inactive lineage A");
    }

    DirectorySnapshot legacy_step_after;
    DirectorySnapshot legacy_private_after;
    if (!SnapshotDirectory(legacy_step, legacy_step_after, error) ||
        !SnapshotDirectory(legacy_private, legacy_private_after, error) ||
        legacy_step_after != legacy_step_before ||
        legacy_private_after != legacy_private_before) {
        return FailAfterServer(
            "legacy flat cache entries were read, changed, or removed");
    }

    const std::string bad_binding_lineage = "bad-binding-lineage";
    const std::string bad_binding_payload = "bad-binding-model";
    AIServerConfig bad_binding_config =
        MakeConfig(root / "bad-binding", port);
    service.SetModel(
        MakeManifest(
            bad_binding_config, bad_binding_lineage, 0,
            bad_binding_payload),
        bad_binding_payload);
    const std::string bad_binding_key = Sha256(bad_binding_lineage);
    const fs::path bad_binding_namespace =
        fs::path(bad_binding_config.model.local_train_dir) /
        "cache/lineages" / bad_binding_key;
    fs::create_directories(bad_binding_namespace);
    const std::string wrong_lineage = "wrong-lineage";
    std::ofstream(bad_binding_namespace / "lineage.json")
        << "{\"schema_version\":1,\"model_lineage_id\":\""
        << wrong_lineage << "\",\"lineage_key\":\""
        << Sha256(wrong_lineage) << "\"}\n";
    DirectorySnapshot bad_binding_before;
    if (!SnapshotDirectory(
            bad_binding_namespace, bad_binding_before, error)) {
        return FailAfterServer(error);
    }
    ModelDistributorClient bad_binding_client(
        bad_binding_config, "aiserver-bad-binding", 1);
    ModelDistributorClient::AvailableRange bad_binding_range;
    ModelDistributorClient::CacheRecoveryFacts bad_binding_facts;
    recovered.clear();
    if (!bad_binding_client.GetAvailableRange(
            "aiserver", bad_binding_range, error) ||
        bad_binding_client.RecoverCache(
            recovered, bad_binding_facts, error)) {
        return FailAfterServer(
            "wrong lineage.json binding did not fail closed");
    }
    DirectorySnapshot bad_binding_after;
    if (!SnapshotDirectory(
            bad_binding_namespace, bad_binding_after, error) ||
        bad_binding_after != bad_binding_before) {
        return FailAfterServer(
            "wrong lineage.json binding was changed during rejection");
    }

    const std::string symlink_lineage = "symlink-lineage";
    const std::string symlink_payload = "symlink-model";
    AIServerConfig symlink_config = MakeConfig(root / "symlink", port);
    service.SetModel(
        MakeManifest(
            symlink_config, symlink_lineage, 0, symlink_payload),
        symlink_payload);
    const fs::path symlink_lineages =
        fs::path(symlink_config.model.local_train_dir) / "cache/lineages";
    const fs::path symlink_outside = root / "symlink-outside";
    fs::create_directories(symlink_lineages);
    fs::create_directories(symlink_outside);
    std::ofstream(symlink_outside / "outside.bin", std::ios::binary)
        << "outside-must-not-change";
    DirectorySnapshot symlink_outside_before;
    if (!SnapshotDirectory(
            symlink_outside, symlink_outside_before, error)) {
        return FailAfterServer(error);
    }
    std::error_code symlink_error;
    fs::create_directory_symlink(
        symlink_outside,
        symlink_lineages / Sha256(symlink_lineage), symlink_error);
    if (symlink_error) {
        return FailAfterServer(
            "cannot create namespace symlink fixture: " +
            symlink_error.message());
    }
    ModelDistributorClient symlink_client(
        symlink_config, "aiserver-symlink", 1);
    ModelDistributorClient::AvailableRange symlink_range;
    ModelDistributorClient::CacheRecoveryFacts symlink_facts;
    recovered.clear();
    if (!symlink_client.GetAvailableRange(
            "aiserver", symlink_range, error) ||
        symlink_client.RecoverCache(
            recovered, symlink_facts, error)) {
        return FailAfterServer(
            "symlink lineage namespace did not fail closed");
    }
    DirectorySnapshot symlink_outside_after;
    if (!SnapshotDirectory(
            symlink_outside, symlink_outside_after, error) ||
        symlink_outside_after != symlink_outside_before) {
        return FailAfterServer(
            "symlink rejection followed or changed the outside directory");
    }

    const std::string invalid_entry_lineage = "invalid-entry-lineage";
    const std::string invalid_entry_payload = "invalid-entry-model";
    AIServerConfig invalid_entry_config =
        MakeConfig(root / "invalid-entry", port);
    service.SetModel(
        MakeManifest(
            invalid_entry_config, invalid_entry_lineage, 0,
            invalid_entry_payload),
        invalid_entry_payload);
    ModelDistributorClient invalid_entry_client(
        invalid_entry_config, "aiserver-invalid-entry", 1);
    ModelDistributorClient::AvailableRange invalid_entry_range;
    ModelDistributorClient::CacheRecoveryFacts invalid_entry_facts;
    recovered.clear();
    if (!invalid_entry_client.GetAvailableRange(
            "aiserver", invalid_entry_range, error) ||
        !invalid_entry_client.RecoverCache(
            recovered, invalid_entry_facts, error)) {
        return FailAfterServer(
            "cannot create valid namespace for invalid-entry check: " +
            error);
    }
    const fs::path invalid_entry_namespace =
        fs::path(invalid_entry_config.model.local_train_dir) /
        "cache/lineages" / Sha256(invalid_entry_lineage);
    std::ofstream(invalid_entry_namespace / "unexpected.bin",
                  std::ios::binary)
        << "must-be-preserved";
    DirectorySnapshot invalid_entry_before;
    if (!SnapshotDirectory(
            invalid_entry_namespace, invalid_entry_before, error)) {
        return FailAfterServer(error);
    }
    ModelDistributorClient invalid_entry_restart(
        invalid_entry_config, "aiserver-invalid-entry-restart", 2);
    ModelDistributorClient::AvailableRange invalid_entry_restart_range;
    ModelDistributorClient::CacheRecoveryFacts invalid_entry_restart_facts;
    recovered.clear();
    if (!invalid_entry_restart.GetAvailableRange(
            "aiserver", invalid_entry_restart_range, error) ||
        invalid_entry_restart.RecoverCache(
            recovered, invalid_entry_restart_facts, error)) {
        return FailAfterServer(
            "unrecognized active namespace entry did not fail closed");
    }
    DirectorySnapshot invalid_entry_after;
    if (!SnapshotDirectory(
            invalid_entry_namespace, invalid_entry_after, error) ||
        invalid_entry_after != invalid_entry_before) {
        return FailAfterServer(
            "invalid namespace rejection changed the active directory");
    }

    server->Shutdown();
    server->Wait();
    fs::remove_all(root);
    return 0;
}
