#include "model/model_manifest.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << std::endl;
        std::exit(1);
    }
}

void WriteManifest(const std::filesystem::path& path,
                   const std::string& checksum,
                   int input_dim,
                   const std::string& contract_version = "0.3.0") {
    std::ofstream stream(path);
    stream << "{"
           << "\"schema_version\":1,"
           << "\"contract_version\":\"" << contract_version << "\","
           << "\"run_id\":\"manifest-run\","
           << "\"model_version\":0,"
           << "\"artifact_uri\":\"file://"
           << (path.parent_path() / "model_v000000.onnx").string()
           << "\","
           << "\"model_file\":\"model_v000000.onnx\","
           << "\"size_bytes\":11,"
           << "\"sha256\":\"" << checksum << "\","
           << "\"input_shape\":[1," << input_dim << "],"
           << "\"action_shape\":[1,9],"
           << "\"value_shape\":[1,1],"
           << "\"seed\":0,"
           << "\"published_ts_ms\":1,"
           << "\"ready\":true"
           << "}\n";
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    fs::path root = fs::temp_directory_path() /
                    ("maze-manifest-test-" +
                     std::to_string(std::rand()));
    fs::path run_dir = root / "manifest-run";
    fs::create_directories(run_dir);
    fs::path model_path = run_dir / "model_v000000.onnx";
    {
        std::ofstream model(model_path, std::ios::binary);
        model << "model-bytes";
    }

    std::string checksum;
    std::string error;
    Require(ComputeFileSha256(model_path.string(), checksum, error),
            "calculate checksum");

    ModelConfig config;
    config.p2p_dir = root.string();
    WriteManifest(run_dir / "manifest.json", checksum, 13);
    ModelManifest manifest;
    Require(LoadModelManifest(
                config, "manifest-run", manifest, error),
            "valid manifest");
    Require(manifest.model_version == 0,
            "model version");
    Require(manifest.sha256 == checksum,
            "model checksum");

    WriteManifest(run_dir / "manifest.json", checksum, 13, "0.2.0");
    Require(!LoadModelManifest(
                config, "manifest-run", manifest, error),
            "old contract version must fail");

    WriteManifest(run_dir / "manifest.json", "invalid", 13);
    Require(!LoadModelManifest(
                config, "manifest-run", manifest, error),
            "checksum mismatch must fail");

    WriteManifest(run_dir / "manifest.json", checksum, 12);
    Require(!LoadModelManifest(
                config, "manifest-run", manifest, error),
            "shape mismatch must fail");

    fs::remove_all(root);
    std::cout << "model_manifest_contract: PASS" << std::endl;
    return 0;
}
