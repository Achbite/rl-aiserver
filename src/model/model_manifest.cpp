#include "model/model_manifest.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <openssl/evp.h>

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace {

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

void FillDigest(const DigestConfig& source, common::ContentDigest* target) {
    target->set_algorithm(common::DIGEST_ALGORITHM_SHA256);
    target->set_hex(source.hex);
}

std::string DeterministicBytes(const google::protobuf::MessageLite& message) {
    std::string output;
    output.resize(message.ByteSizeLong());
    google::protobuf::io::ArrayOutputStream array(
        output.data(), static_cast<int>(output.size()));
    google::protobuf::io::CodedOutputStream coded(&array);
    coded.SetSerializationDeterministic(true);
    if (!message.SerializeToCodedStream(&coded) || coded.HadError()) {
        return "";
    }
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

bool WriteAll(int descriptor, const char* data, std::size_t size) {
    std::size_t written = 0;
    while (written < size) {
        const ssize_t count =
            ::write(descriptor, data + written, size - written);
        if (count <= 0) return false;
        written += static_cast<std::size_t>(count);
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
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context || EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1) {
        if (context) EVP_MD_CTX_free(context);
        error = "cannot initialise model SHA-256";
        return false;
    }
    std::array<char, 1024 * 1024> buffer{};
    bool ok = true;
    while (stream.good()) {
        stream.read(buffer.data(), buffer.size());
        const auto count = stream.gcount();
        if (count > 0 &&
            EVP_DigestUpdate(context, buffer.data(),
                             static_cast<std::size_t>(count)) != 1) {
            ok = false;
            break;
        }
    }
    if (stream.bad()) ok = false;
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int size = 0;
    if (ok) ok = EVP_DigestFinal_ex(context, digest.data(), &size) == 1;
    EVP_MD_CTX_free(context);
    if (!ok) {
        error = "cannot hash model file: " + path;
        return false;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < size; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    checksum = output.str();
    error.clear();
    return true;
}

bool ValidateModelManifest(const AIServerConfig& config,
                           const training::ModelArtifactManifest& source,
                           std::optional<ModelStep> expected_step,
                           std::string& error) {
    common::ContentDigest expected_training_contract;
    FillDigest(config.training_contract.canonical_digest,
               &expected_training_contract);
    if (source.training_contract_digest().SerializeAsString() !=
            expected_training_contract.SerializeAsString()) {
        error = "model manifest contract identity is invalid";
        return false;
    }
    if (source.identity().model_lineage_id().empty() ||
        !source.identity().has_model_step() ||
        (expected_step.has_value() &&
         source.identity().model_step() != *expected_step) ||
        !IsSha256(source.identity().artifact_digest()) ||
        !IsSha256(source.identity().manifest_digest()) ||
        !IsSha256(source.training_config_digest()) ||
        source.size_bytes() <= 0 ||
        source.published_at_unix_ms() <= 0) {
        error = "model manifest identity or artifact metadata is invalid";
        return false;
    }

    const auto& profile = source.rollout_estimator_profile();
    training::RolloutEstimatorProfile canonical_profile(profile);
    canonical_profile.clear_profile_digest();
    const std::string canonical_profile_digest =
        Sha256Bytes(DeterministicBytes(canonical_profile));
    if (!std::isfinite(profile.gamma()) || profile.gamma() < 0.0 ||
        profile.gamma() > 1.0 ||
        !std::isfinite(profile.gae_lambda()) ||
        profile.gae_lambda() < 0.0 || profile.gae_lambda() > 1.0 ||
        profile.tmax() == 0 ||
        !IsSha256(profile.profile_digest()) ||
        canonical_profile_digest.empty() ||
        canonical_profile_digest != profile.profile_digest().hex()) {
        error = "rollout estimator profile is invalid or non-canonical";
        return false;
    }

    training::ModelArtifactManifest digest_source(source);
    digest_source.mutable_identity()->clear_manifest_digest();
    const std::string digest = Sha256Bytes(DeterministicBytes(digest_source));
    if (digest.empty() || digest != source.identity().manifest_digest().hex()) {
        error = "model manifest digest mismatch";
        return false;
    }
    error.clear();
    return true;
}

void AssignModelManifest(const training::ModelArtifactManifest& source,
                         const std::string& model_path,
                         ModelManifest& destination) {
    destination = ModelManifest{};
    destination.wire = source;
    destination.model_path = model_path;
}

bool LoadModelManifestFile(const AIServerConfig& config,
                           const std::string& manifest_file_path,
                           ModelManifest& manifest,
                           std::string& error) {
    const std::filesystem::path manifest_path = manifest_file_path;
    if (manifest_path.filename() != kModelManifestFile) {
        error = "model manifest must use the canonical protobuf file name";
        return false;
    }
    std::ifstream stream(manifest_path, std::ios::binary);
    if (!stream.is_open()) {
        error = "manifest not found: " + manifest_path.string();
        return false;
    }
    training::ModelArtifactManifest wire;
    if (!wire.ParseFromIstream(&stream) || stream.bad() ||
        !ValidateModelManifest(config, wire, std::nullopt, error)) {
        if (error.empty()) error = "invalid protobuf model manifest";
        return false;
    }

    const std::filesystem::path model_path =
        manifest_path.parent_path() / kPublishedModelFile;
    std::error_code filesystem_error;
    const auto actual_size =
        std::filesystem::file_size(model_path, filesystem_error);
    if (filesystem_error ||
        actual_size != static_cast<std::uintmax_t>(wire.size_bytes())) {
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
    error.clear();
    return true;
}

bool WriteModelManifestFile(
    const training::ModelArtifactManifest& manifest,
    const std::string& manifest_file_path,
    std::string& error) {
    const std::filesystem::path destination(manifest_file_path);
    if (destination.filename() != kModelManifestFile) {
        error = "model manifest must use the canonical protobuf file name";
        return false;
    }
    const std::string payload = DeterministicBytes(manifest);
    if (payload.empty()) {
        error = "cannot serialize protobuf model manifest";
        return false;
    }
    const std::filesystem::path temporary =
        destination.string() + ".tmp." + std::to_string(::getpid());
    const int descriptor = ::open(
        temporary.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
    if (descriptor < 0) {
        error = "cannot open temporary model manifest for writing";
        return false;
    }
    bool ok = WriteAll(descriptor, payload.data(), payload.size());
    if (ok) ok = ::fsync(descriptor) == 0;
    if (::close(descriptor) != 0) ok = false;
    if (ok) {
        std::error_code rename_error;
        std::filesystem::rename(temporary, destination, rename_error);
        ok = !rename_error;
    }
    if (!ok) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary, cleanup_error);
        error = "cannot durably publish protobuf model manifest";
        return false;
    }
    error.clear();
    return true;
}
