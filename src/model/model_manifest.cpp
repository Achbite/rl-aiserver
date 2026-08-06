#include "model/model_manifest.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>
#include <openssl/evp.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace {

using google::protobuf::Struct;
using google::protobuf::Value;

const Value* FindField(const Struct& object, const std::string& name) {
    const auto it = object.fields().find(name);
    return it == object.fields().end() ? nullptr : &it->second;
}

bool ReadStruct(const Struct& object, const std::string& name,
                const Struct*& value, std::string& error) {
    const Value* field = FindField(object, name);
    if (!field || field->kind_case() != Value::kStructValue) {
        error = "manifest field '" + name + "' must be an object";
        return false;
    }
    value = &field->struct_value();
    return true;
}

bool ReadString(const Struct& object, const std::string& name,
                std::string& value, std::string& error) {
    const Value* field = FindField(object, name);
    if (!field || field->kind_case() != Value::kStringValue) {
        error = "manifest field '" + name + "' must be a string";
        return false;
    }
    value = field->string_value();
    return true;
}

bool ReadInteger(const Struct& object, const std::string& name,
                 int64_t& value, std::string& error) {
    const Value* field = FindField(object, name);
    if (!field || field->kind_case() != Value::kNumberValue) {
        error = "manifest field '" + name + "' must be an integer";
        return false;
    }
    value = static_cast<int64_t>(field->number_value());
    if (static_cast<double>(value) != field->number_value()) {
        error = "manifest field '" + name + "' must be an integer";
        return false;
    }
    return true;
}

bool ReadBoolean(const Struct& object, const std::string& name,
                 bool& value, std::string& error) {
    const Value* field = FindField(object, name);
    if (!field || field->kind_case() != Value::kBoolValue) {
        error = "manifest field '" + name + "' must be a boolean";
        return false;
    }
    value = field->bool_value();
    return true;
}

bool ReadShape(const Struct& object, const std::string& name,
               google::protobuf::RepeatedField<int64_t>* shape,
               std::string& error) {
    const Value* field = FindField(object, name);
    if (!field || field->kind_case() != Value::kListValue) {
        error = "manifest field '" + name + "' must be an array";
        return false;
    }
    shape->Clear();
    for (const auto& item : field->list_value().values()) {
        if (item.kind_case() != Value::kNumberValue) {
            error = "manifest shape '" + name + "' must contain integers";
            return false;
        }
        const int64_t dimension = static_cast<int64_t>(item.number_value());
        if (static_cast<double>(dimension) != item.number_value()) {
            error = "manifest shape '" + name + "' must contain integers";
            return false;
        }
        shape->Add(dimension);
    }
    return true;
}

bool IsSha256(const common::ContentDigest& digest) {
    if (digest.algorithm() != common::DIGEST_ALGORITHM_SHA256 ||
        digest.hex().size() != 64) {
        return false;
    }
    for (const char character : digest.hex()) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

void SetDigest(const std::string& value, common::ContentDigest* digest) {
    digest->set_algorithm(common::DIGEST_ALGORITHM_SHA256);
    digest->set_hex(value);
}

bool ReadDigest(const Struct& object, const std::string& name,
                common::ContentDigest* digest, std::string& error) {
    std::string value;
    if (!ReadString(object, name, value, error)) return false;
    SetDigest(value, digest);
    if (!IsSha256(*digest)) {
        error = "manifest digest '" + name + "' is invalid";
        return false;
    }
    return true;
}

bool ReadSchema(const Struct& object, const std::string& name,
                common::SchemaIdentity* schema, std::string& error) {
    const Struct* source = nullptr;
    int64_t version = 0;
    std::string schema_id;
    if (!ReadStruct(object, name, source, error) ||
        !ReadString(*source, "schema_id", schema_id, error) ||
        !ReadInteger(*source, "schema_version", version, error) ||
        !ReadDigest(*source, "canonical_digest",
                    schema->mutable_canonical_digest(), error)) {
        return false;
    }
    if (schema_id.empty() || version <= 0 || version > UINT32_MAX) {
        error = "manifest schema identity is invalid: " + name;
        return false;
    }
    schema->set_schema_id(schema_id);
    schema->set_schema_version(static_cast<uint32_t>(version));
    return true;
}

bool ReadContract(const Struct& object, common::ContractIdentity* contract,
                  std::string& error) {
    const Struct* source = nullptr;
    if (!ReadStruct(object, "contract", source, error) ||
        !ReadString(*source, "package_name", *contract->mutable_package_name(), error) ||
        !ReadString(*source, "package_version", *contract->mutable_package_version(), error) ||
        !ReadDigest(*source, "source_digest", contract->mutable_source_digest(), error) ||
        !ReadDigest(*source, "artifact_digest", contract->mutable_artifact_digest(), error) ||
        !ReadString(*source, "platform", *contract->mutable_platform(), error) ||
        !ReadString(*source, "generator_identity", *contract->mutable_generator_identity(), error)) {
        return false;
    }
    return true;
}

bool ReadTrainingSemantics(const Struct& object,
                           training::TrainingSemanticsIdentity* semantics,
                           std::string& error) {
    const Struct* source = nullptr;
    if (!ReadStruct(object, "training_semantics", source, error) ||
        !ReadString(*source, "training_contract_id",
                    *semantics->mutable_training_contract_id(), error) ||
        !ReadSchema(*source, "observation_schema",
                    semantics->mutable_observation_schema(), error) ||
        !ReadSchema(*source, "action_schema",
                    semantics->mutable_action_schema(), error) ||
        !ReadSchema(*source, "reward_schema",
                    semantics->mutable_reward_schema(), error) ||
        !ReadString(*source, "policy_distribution_schema_id",
                    *semantics->mutable_policy_distribution_schema_id(), error) ||
        !ReadString(*source, "model_architecture_id",
                    *semantics->mutable_model_architecture_id(), error) ||
        !ReadDigest(*source, "semantics_digest",
                    semantics->mutable_semantics_digest(), error)) {
        return false;
    }
    return true;
}

bool ParseManifestDocument(const Struct& document,
                           training::ModelArtifactManifest& manifest,
                           std::string& error) {
    const Struct* identity = nullptr;
    int64_t schema_version = 0;
    int64_t model_version = 0;
    int64_t size_bytes = 0;
    int64_t seed = 0;
    int64_t train_updates = 0;
    int64_t trained_samples = 0;
    int64_t published_at = 0;
    bool ready = false;
    std::string lineage;
    if (!ReadInteger(document, "manifest_schema_version", schema_version, error) ||
        !ReadContract(document, manifest.mutable_contract(), error) ||
        !ReadStruct(document, "identity", identity, error) ||
        !ReadString(*identity, "model_lineage_id", lineage, error) ||
        !ReadInteger(*identity, "model_version", model_version, error) ||
        !ReadDigest(*identity, "artifact_digest",
                    manifest.mutable_identity()->mutable_artifact_digest(), error) ||
        !ReadDigest(*identity, "manifest_digest",
                    manifest.mutable_identity()->mutable_manifest_digest(), error) ||
        !ReadSchema(document, "observation_schema",
                    manifest.mutable_observation_schema(), error) ||
        !ReadSchema(document, "action_schema",
                    manifest.mutable_action_schema(), error) ||
        !ReadString(document, "model_architecture_id",
                    *manifest.mutable_model_architecture_id(), error) ||
        !ReadString(document, "tensor_dtype",
                    *manifest.mutable_tensor_dtype(), error) ||
        !ReadShape(document, "input_shape", manifest.mutable_input_shape(), error) ||
        !ReadShape(document, "action_shape", manifest.mutable_action_shape(), error) ||
        !ReadShape(document, "value_shape", manifest.mutable_value_shape(), error) ||
        !ReadString(document, "artifact_uri", *manifest.mutable_artifact_uri(), error) ||
        !ReadString(document, "model_file", *manifest.mutable_model_file(), error) ||
        !ReadInteger(document, "size_bytes", size_bytes, error) ||
        !ReadInteger(document, "seed", seed, error) ||
        !ReadInteger(document, "train_updates", train_updates, error) ||
        !ReadInteger(document, "trained_samples", trained_samples, error) ||
        !ReadDigest(document, "training_config_digest",
                    manifest.mutable_training_config_digest(), error) ||
        !ReadTrainingSemantics(document,
                               manifest.mutable_training_semantics(), error) ||
        !ReadInteger(document, "published_at_unix_ms", published_at, error) ||
        !ReadBoolean(document, "ready", ready, error)) {
        return false;
    }
    if (schema_version < 0 || schema_version > UINT32_MAX ||
        model_version < 0) {
        error = "manifest schema or model version is invalid";
        return false;
    }
    manifest.set_manifest_schema_version(static_cast<uint32_t>(schema_version));
    manifest.mutable_identity()->set_model_lineage_id(lineage);
    manifest.mutable_identity()->set_model_version(
        static_cast<uint64_t>(model_version));
    manifest.set_size_bytes(size_bytes);
    manifest.set_seed(seed);
    manifest.set_train_updates(train_updates);
    manifest.set_trained_samples(trained_samples);
    manifest.set_published_at_unix_ms(published_at);
    manifest.set_ready(ready);
    return true;
}

std::string DeterministicBytes(const google::protobuf::MessageLite& message) {
    std::string output;
    output.resize(message.ByteSizeLong());
    google::protobuf::io::ArrayOutputStream array(
        output.data(), static_cast<int>(output.size()));
    google::protobuf::io::CodedOutputStream coded(&array);
    coded.SetSerializationDeterministic(true);
    message.SerializeToCodedStream(&coded);
    output.resize(static_cast<std::size_t>(coded.ByteCount()));
    return output;
}

std::string Sha256Bytes(const std::string& payload) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context) return "";
    bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
              EVP_DigestUpdate(context, payload.data(), payload.size()) == 1;
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int size = 0;
    if (ok) ok = EVP_DigestFinal_ex(context, digest.data(), &size) == 1;
    EVP_MD_CTX_free(context);
    if (!ok) return "";
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < size; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    return output.str();
}

void FillDigest(const DigestConfig& source, common::ContentDigest* target) {
    target->set_algorithm(common::DIGEST_ALGORITHM_SHA256);
    target->set_hex(source.hex);
}

void FillSchema(const SchemaConfig& source, common::SchemaIdentity* target) {
    target->set_schema_id(source.schema_id);
    target->set_schema_version(source.schema_version);
    FillDigest(source.canonical_digest, target->mutable_canonical_digest());
}

common::ContractIdentity ExpectedContract(const AIServerConfig& config) {
    common::ContractIdentity result;
    result.set_package_name(config.contract.package_name);
    result.set_package_version(config.contract.package_version);
    FillDigest(config.contract.source_digest, result.mutable_source_digest());
    FillDigest(config.contract.artifact_digest, result.mutable_artifact_digest());
    result.set_platform(config.contract.platform);
    result.set_generator_identity(config.contract.generator_identity);
    return result;
}

training::TrainingSemanticsIdentity ExpectedSemantics(
    const AIServerConfig& config) {
    training::TrainingSemanticsIdentity result;
    result.set_training_contract_id(
        config.training_semantics.training_contract_id);
    FillSchema(config.training_semantics.observation_schema,
               result.mutable_observation_schema());
    FillSchema(config.training_semantics.action_schema,
               result.mutable_action_schema());
    FillSchema(config.training_semantics.reward_schema,
               result.mutable_reward_schema());
    result.set_policy_distribution_schema_id(
        config.training_semantics.policy_distribution_schema_id);
    result.set_model_architecture_id(
        config.training_semantics.model_architecture_id);
    FillDigest(config.training_semantics.semantics_digest,
               result.mutable_semantics_digest());
    return result;
}

bool ShapeEquals(const google::protobuf::RepeatedField<int64_t>& actual,
                 std::initializer_list<int64_t> expected) {
    if (actual.size() != static_cast<int>(expected.size())) return false;
    int index = 0;
    for (const int64_t value : expected) {
        if (actual.Get(index++) != value) return false;
    }
    return true;
}

}  // namespace

bool ComputeFileSha256(const std::string& path,
                       std::string& checksum,
                       std::string& error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        error = "cannot open model file: " + path;
        return false;
    }
    std::ostringstream payload;
    payload << stream.rdbuf();
    if (stream.bad()) {
        error = "cannot read model file: " + path;
        return false;
    }
    checksum = Sha256Bytes(payload.str());
    if (checksum.empty()) {
        error = "failed to calculate SHA-256: " + path;
        return false;
    }
    return true;
}

bool ValidateModelManifest(const AIServerConfig& config,
                           const training::ModelArtifactManifest& source,
                           int expected_version,
                           std::string& error) {
    if (source.manifest_schema_version() != 1 || !source.ready() ||
        source.contract().SerializeAsString() !=
            ExpectedContract(config).SerializeAsString() ||
        source.training_semantics().SerializeAsString() !=
            ExpectedSemantics(config).SerializeAsString()) {
        error = "model manifest contract or training semantics mismatch";
        return false;
    }
    if (source.identity().model_lineage_id() !=
            config.model.expected_model_lineage_id ||
        (expected_version >= 0 &&
         source.identity().model_version() !=
             static_cast<uint64_t>(expected_version)) ||
        !IsSha256(source.identity().artifact_digest()) ||
        !IsSha256(source.identity().manifest_digest()) ||
        !IsSha256(source.training_config_digest()) ||
        source.size_bytes() <= 0 || source.train_updates() < 0 ||
        source.trained_samples() < 0) {
        error = "model manifest identity or counters are invalid";
        return false;
    }
    if (source.observation_schema().SerializeAsString() !=
            ExpectedSemantics(config).observation_schema().SerializeAsString() ||
        source.action_schema().SerializeAsString() !=
            ExpectedSemantics(config).action_schema().SerializeAsString() ||
        source.model_architecture_id() != config.model.model_architecture_id ||
        source.tensor_dtype() != config.model.tensor_dtype ||
        !ShapeEquals(source.input_shape(), {1, config.model.expected_obs_dim}) ||
        !ShapeEquals(source.action_shape(), {1, config.model.expected_action_dim}) ||
        !ShapeEquals(source.value_shape(), {1, 1})) {
        error = "model manifest schema, dtype or shape mismatch";
        return false;
    }
    if (source.model_file().empty() ||
        std::filesystem::path(source.model_file()).filename() !=
            std::filesystem::path(source.model_file())) {
        error = "model_file must be a file name";
        return false;
    }
    training::ModelArtifactManifest digest_source = source;
    digest_source.mutable_identity()->clear_manifest_digest();
    const std::string digest = Sha256Bytes(DeterministicBytes(digest_source));
    if (digest.empty() || digest != source.identity().manifest_digest().hex()) {
        error = "model manifest digest mismatch";
        return false;
    }
    return true;
}

void AssignModelManifest(const training::ModelArtifactManifest& source,
                         const std::string& model_path,
                         ModelManifest& destination) {
    destination = ModelManifest{};
    destination.wire = source;
    destination.schema_version =
        static_cast<int>(source.manifest_schema_version());
    destination.contract_version = source.contract().package_version();
    destination.model_lineage_id = source.identity().model_lineage_id();
    destination.model_version =
        static_cast<int>(source.identity().model_version());
    destination.sha256 = source.identity().artifact_digest().hex();
    destination.manifest_digest = source.identity().manifest_digest().hex();
    destination.artifact_uri = source.artifact_uri();
    destination.model_file = source.model_file();
    destination.size_bytes = source.size_bytes();
    destination.input_shape.assign(
        source.input_shape().begin(), source.input_shape().end());
    destination.action_shape.assign(
        source.action_shape().begin(), source.action_shape().end());
    destination.value_shape.assign(
        source.value_shape().begin(), source.value_shape().end());
    destination.seed = source.seed();
    destination.published_ts_ms = source.published_at_unix_ms();
    destination.observation_schema_id =
        source.observation_schema().schema_id();
    destination.action_schema_id = source.action_schema().schema_id();
    destination.model_architecture_id = source.model_architecture_id();
    destination.tensor_dtype = source.tensor_dtype();
    destination.train_updates = source.train_updates();
    destination.trained_samples = source.trained_samples();
    destination.ready = source.ready();
    destination.model_path = model_path;
}

bool LoadModelManifest(const AIServerConfig& config,
                       ModelManifest& manifest,
                       std::string& error) {
    const std::filesystem::path manifest_path =
        std::filesystem::path(config.model.local_train_dir) /
        "active" / config.model.manifest_name;
    return LoadModelManifestFile(
        config, manifest_path.string(), manifest, error);
}

bool LoadModelManifestFile(const AIServerConfig& config,
                           const std::string& manifest_file_path,
                           ModelManifest& manifest,
                           std::string& error) {
    const std::filesystem::path manifest_path = manifest_file_path;
    std::ifstream stream(manifest_path);
    if (!stream.is_open()) {
        error = "manifest not found: " + manifest_path.string();
        return false;
    }
    std::ostringstream content;
    content << stream.rdbuf();
    Struct document;
    const auto status = google::protobuf::util::JsonStringToMessage(
        content.str(), &document);
    if (!status.ok()) {
        error = "invalid manifest JSON: " + status.ToString();
        return false;
    }
    training::ModelArtifactManifest wire;
    if (!ParseManifestDocument(document, wire, error) ||
        !ValidateModelManifest(config, wire, -1, error)) {
        return false;
    }
    const std::filesystem::path model_path =
        manifest_path.parent_path() / wire.model_file();
    std::error_code filesystem_error;
    const auto actual_size =
        std::filesystem::file_size(model_path, filesystem_error);
    if (filesystem_error ||
        static_cast<int64_t>(actual_size) != wire.size_bytes()) {
        error = "model size does not match manifest";
        return false;
    }
    std::string actual_digest;
    if (!ComputeFileSha256(model_path.string(), actual_digest, error) ||
        actual_digest != wire.identity().artifact_digest().hex()) {
        if (error.empty()) error = "model artifact digest mismatch";
        return false;
    }
    AssignModelManifest(wire, model_path.string(), manifest);
    manifest.manifest_path = manifest_path.string();
    return true;
}
