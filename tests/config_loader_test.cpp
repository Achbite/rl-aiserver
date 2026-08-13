#include "config/config_loader.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

namespace {

void Require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::ostringstream content;
    content << input.rdbuf();
    Require(static_cast<bool>(input) || input.eof(), "cannot read config");
    return content.str();
}

std::string ReplaceOnce(std::string value,
                        const std::string& from,
                        const std::string& to) {
    const auto position = value.find(from);
    Require(position != std::string::npos, "config fixture token missing");
    value.replace(position, from.size(), to);
    return value;
}

std::string ReplaceManifestChecksum(std::string manifest,
                                    const std::string& filename) {
    const std::string prefix = "\"" + filename + "\": \"";
    const auto prefix_position = manifest.find(prefix);
    Require(prefix_position != std::string::npos,
            "manifest fixture checksum is missing");
    const auto checksum_position = prefix_position + prefix.size();
    Require(checksum_position + 64 <= manifest.size(),
            "manifest fixture checksum is truncated");
    manifest.replace(checksum_position, 64, std::string(64, '0'));
    return manifest;
}

bool LoadDocument(const std::filesystem::path& root,
                  const std::string& name,
                  const std::string& content) {
    const auto path = root / name;
    std::ofstream output(path, std::ios::trunc);
    output << content;
    output.close();
    AIServerConfig config;
    return LoadServerConfig(path.string(), config);
}

}  // namespace

int main(int argc, char* argv[]) {
    Require(argc == 2, "config path argument is required");
    const auto config_path = std::filesystem::weakly_canonical(argv[1]);
    const auto repository_root = config_path.parent_path().parent_path();
    const auto root = std::filesystem::temp_directory_path() /
                      ("rl-aiserver-config-test-" +
                       std::to_string(::getpid()));
    const auto snapshot = root / "proto";
    const auto snapshot_schemas = snapshot / "schemas";
    std::filesystem::create_directories(snapshot_schemas);
    std::filesystem::copy_file(
        repository_root / "proto/manifest.json",
        snapshot / "manifest.json");
    std::filesystem::copy_file(
        repository_root / "proto/schemas/maze.metrics.v2.json",
        snapshot_schemas / "maze.metrics.v2.json");
    std::filesystem::copy_file(
        repository_root / "proto/schemas/maze.metrics.v2.sha256",
        snapshot_schemas / "maze.metrics.v2.sha256");
    const std::string valid = ReplaceOnce(
        ReadFile(config_path), "../proto/schemas/maze.metrics.v2.json",
        (snapshot_schemas / "maze.metrics.v2.json").string());

    Require(LoadDocument(root, "valid.yaml", valid),
            "valid config was rejected");
    {
        std::ofstream catalog(
            snapshot_schemas / "maze.metrics.v2.json", std::ios::app);
        catalog << ' ';
    }
    Require(!LoadDocument(root, "tampered-catalog.yaml", valid),
            "tampered metric schema catalog did not fail closed");
    std::filesystem::copy_file(
        repository_root / "proto/schemas/maze.metrics.v2.json",
        snapshot_schemas / "maze.metrics.v2.json",
        std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream digest(
            snapshot_schemas / "maze.metrics.v2.sha256", std::ios::trunc);
        digest << std::string(64, '0') << '\n';
    }
    Require(!LoadDocument(root, "tampered-digest.yaml", valid),
            "tampered metric schema digest did not fail closed");
    std::filesystem::copy_file(
        repository_root / "proto/schemas/maze.metrics.v2.sha256",
        snapshot_schemas / "maze.metrics.v2.sha256",
        std::filesystem::copy_options::overwrite_existing);
    const std::string valid_manifest = ReadFile(
        repository_root / "proto/manifest.json");
    {
        std::ofstream manifest(snapshot / "manifest.json", std::ios::trunc);
        manifest << ReplaceManifestChecksum(valid_manifest, "common.proto");
    }
    Require(!LoadDocument(root, "tampered-file-table.yaml", valid),
            "tampered artifact file table did not fail closed");
    {
        std::ofstream manifest(snapshot / "manifest.json", std::ios::trunc);
        manifest << valid_manifest;
    }
    Require(!LoadDocument(
                root, "bad-revision.yaml",
                ReplaceOnce(valid, "  task_revision: 2\n",
                            "  task_revision: 2suffix\n")),
            "malformed task revision did not fail closed");
    Require(!LoadDocument(
                root, "bad-failure-budget.yaml",
                ReplaceOnce(valid, "  timeout_penalty: -2.0",
                            "  timeout_penalty: -1.0")),
            "unsafe TIME_LIMIT failure budget did not fail closed");
    Require(!LoadDocument(
                root, "bad-temperature.yaml",
                ReplaceOnce(valid, "  training_temperature: 1.0",
                            "  training_temperature: nan")),
            "non-finite training temperature did not fail closed");
    Require(!LoadDocument(
                root, "bad-mode.yaml",
                ReplaceOnce(valid, "  run_mode: 1",
                            "  run_mode: unexpected")),
            "unknown workload did not fail closed");
    const char* retired_curriculum_fields[] = {
        "stage_8x_sample_budget",
        "stage_4x_sample_budget",
        "stage_2x_sample_budget",
        "evaluation_interval_samples",
        "evaluation_episodes_per_round",
        "stage_8x_success_threshold",
        "stage_4x_success_threshold",
        "stage_2x_success_threshold",
        "final_path_ratio_median_limit",
        "final_path_ratio_p95_limit",
    };
    for (const char* field : retired_curriculum_fields) {
        Require(!LoadDocument(
                    root, std::string("retired-") + field + ".yaml",
                    valid + "\ncurriculum:\n  " + field + ": 1\n"),
                "retired curriculum/evaluation field did not fail closed");
    }
    Require(!LoadDocument(
                root, "retired-training-budget.yaml",
                valid + "\ntask:\n  training_sample_budget: 1000000\n"),
            "retired training sample budget did not fail closed");

    ::setenv("RL_SAMPLE_FRAGMENT_SIZE", "128suffix", 1);
    Require(!LoadDocument(root, "bad-env.yaml", valid),
            "malformed environment override did not fail closed");
    ::unsetenv("RL_SAMPLE_FRAGMENT_SIZE");

    std::error_code error;
    std::filesystem::remove_all(root, error);
    return 0;
}
