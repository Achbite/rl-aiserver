#include "model/model_manifest.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>

namespace {

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

bool ValidateModelManifest(const AIServerConfig& config,
                           const training::ModelArtifactManifest& source,
                           std::optional<ModelStep> expected_step,
                           std::string& error) {
    if (source.identity().model_lineage_id().empty() ||
        !source.identity().has_model_step() ||
        (expected_step.has_value() &&
         source.identity().model_step() != *expected_step) ||
        source.size_bytes() <= 0 ||
        source.published_at_unix_ms() <= 0) {
        error = "model manifest identity or artifact metadata is invalid";
        return false;
    }
    if (!std::isfinite(config.rollout.gamma) ||
        config.rollout.gamma < 0.0 || config.rollout.gamma > 1.0 ||
        !std::isfinite(config.rollout.gae_lambda) ||
        config.rollout.gae_lambda < 0.0 ||
        config.rollout.gae_lambda > 1.0 || config.rollout.tmax == 0) {
        error = "AIServer rollout configuration is invalid";
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
    AssignModelManifest(wire, model_path.string(), manifest);
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
