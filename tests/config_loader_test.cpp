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

bool LoadDocumentWithOverrides(
    const std::filesystem::path& root,
    const std::string& name,
    const std::string& content,
    const AIServerConfigOverrides& overrides,
    AIServerConfig* loaded_config = nullptr,
    std::string* load_error = nullptr) {
    const auto path = root / name;
    std::ofstream output(path, std::ios::trunc);
    output << content;
    output.close();
    AIServerConfig config;
    AIServerConfigLoadReport report;
    std::string error;
    const bool loaded = LoadServerConfig(
        path.string(), overrides, config, report, error);
    if (loaded_config) *loaded_config = config;
    if (load_error) *load_error = error;
    return loaded;
}

bool LoadDocument(const std::filesystem::path& root,
                  const std::string& name,
                  const std::string& content) {
    AIServerConfigOverrides overrides;
    overrides.workload = aiserver_mode::kTraining;
    return LoadDocumentWithOverrides(
        root, name, content, overrides);
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
        repository_root / "proto/schemas/maze.metrics.v3.json",
        snapshot_schemas / "maze.metrics.v3.json");
    std::filesystem::copy_file(
        repository_root / "proto/schemas/maze.metrics.v3.sha256",
        snapshot_schemas / "maze.metrics.v3.sha256");
    const std::string valid = ReplaceOnce(
        ReadFile(config_path), "../proto/schemas/maze.metrics.v3.json",
        (snapshot_schemas / "maze.metrics.v3.json").string());

    AIServerConfig default_task;
    AIServerConfigOverrides training_override;
    training_override.workload = aiserver_mode::kTraining;
    Require(LoadDocumentWithOverrides(
                root, "valid.yaml", valid, training_override,
                &default_task) &&
                default_task.task.task_config_digest.hex.size() == 64 &&
                default_task.sample_distributor.recovery_timeout_ms == 30000,
            "valid config or computed task identity was rejected");

    const auto evaluation_dir = root / "models/eval/0000000";
    std::filesystem::create_directories(evaluation_dir);
    const auto evaluation_model = evaluation_dir / "SaveModel.onnx";
    {
        std::ofstream output(evaluation_model, std::ios::binary);
        output << "onnx";
    }
    const std::string evaluation_config = ReplaceOnce(
        valid, "../models/eval/0000000/SaveModel.onnx",
        "models/eval/0000000/SaveModel.onnx");
    AIServerConfig loaded_evaluation;
    AIServerConfigOverrides evaluation_override;
    evaluation_override.workload = aiserver_mode::kEvaluation;
    Require(LoadDocumentWithOverrides(
                root, "evaluation-relative.yaml", evaluation_config,
                evaluation_override, &loaded_evaluation) &&
                loaded_evaluation.server.run_mode ==
                    aiserver_mode::kEvaluation &&
                loaded_evaluation.model.evaluation_model_path ==
                    std::filesystem::absolute(evaluation_model)
                        .lexically_normal().string(),
            "evaluation config path was not resolved relative to config");

    const auto cli_model = root / "cli/SaveModel.onnx";
    std::filesystem::create_directories(cli_model.parent_path());
    {
        std::ofstream output(cli_model, std::ios::binary);
        output << "onnx";
    }
    AIServerConfigOverrides evaluation_cli;
    evaluation_cli.workload = aiserver_mode::kEvaluation;
    evaluation_cli.evaluation_model_path = "cli/SaveModel.onnx";
    Require(LoadDocumentWithOverrides(
                root, "evaluation-cli.yaml", valid, evaluation_cli,
                &loaded_evaluation) &&
                loaded_evaluation.model.evaluation_model_path ==
                    std::filesystem::absolute(cli_model)
                        .lexically_normal().string(),
            "relative CLI evaluation model was not resolved against config");

    AIServerConfigOverrides invalid_training_cli;
    invalid_training_cli.workload = aiserver_mode::kTraining;
    invalid_training_cli.evaluation_model_path = cli_model.string();
    std::string load_error;
    Require(!LoadDocumentWithOverrides(
                root, "training-with-evaluation-model.yaml", valid,
                invalid_training_cli, nullptr, &load_error) &&
                load_error.find("invalid for the training workload") !=
                    std::string::npos,
            "training accepted an evaluation-only model override");

    const auto wrong_name = root / "models/eval/model.onnx";
    {
        std::ofstream output(wrong_name, std::ios::binary);
        output << "onnx";
    }
    Require(!LoadDocumentWithOverrides(
                root, "evaluation-wrong-name.yaml",
                ReplaceOnce(evaluation_config,
                            "models/eval/0000000/SaveModel.onnx",
                            "models/eval/model.onnx"),
                evaluation_override),
            "evaluation accepted a model not named SaveModel.onnx");

    const auto symlink_dir = root / "models/symlink";
    std::filesystem::create_directories(symlink_dir);
    std::filesystem::create_symlink(
        evaluation_model, symlink_dir / "SaveModel.onnx");
    Require(!LoadDocumentWithOverrides(
                root, "evaluation-symlink.yaml",
                ReplaceOnce(evaluation_config,
                            "models/eval/0000000/SaveModel.onnx",
                            "models/symlink/SaveModel.onnx"),
                evaluation_override),
            "evaluation accepted a symbolic-link model");

    Require(!LoadDocumentWithOverrides(
                root, "evaluation-missing.yaml",
                ReplaceOnce(evaluation_config,
                            "models/eval/0000000/SaveModel.onnx",
                            "models/missing/SaveModel.onnx"),
                evaluation_override),
            "evaluation accepted a missing model");

    {
        std::ofstream catalog(
            snapshot_schemas / "maze.metrics.v3.json", std::ios::app);
        catalog << ' ';
    }
    Require(!LoadDocument(root, "tampered-catalog.yaml", valid),
            "tampered metric schema catalog did not fail closed");
    std::filesystem::copy_file(
        repository_root / "proto/schemas/maze.metrics.v3.json",
        snapshot_schemas / "maze.metrics.v3.json",
        std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream digest(
            snapshot_schemas / "maze.metrics.v3.sha256", std::ios::trunc);
        digest << std::string(64, '0') << '\n';
    }
    Require(!LoadDocument(root, "tampered-digest.yaml", valid),
            "tampered metric schema digest did not fail closed");
    std::filesystem::copy_file(
        repository_root / "proto/schemas/maze.metrics.v3.sha256",
        snapshot_schemas / "maze.metrics.v3.sha256",
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
                ReplaceOnce(valid, "  task_revision: 3\n",
                            "  task_revision: 3suffix\n")),
            "malformed task revision did not fail closed");
    Require(!LoadDocument(
                root, "bad-temperature.yaml",
                ReplaceOnce(valid, "  training_temperature: 1.0",
                            "  training_temperature: nan")),
            "non-finite training temperature did not fail closed");
    Require(!LoadDocument(
                root, "bad-mode.yaml",
                ReplaceOnce(valid, "  run_mode: \"training\"",
                            "  run_mode: \"unexpected\"")),
            "unknown workload did not fail closed");
    Require(!LoadDocument(
                root, "platform-id.yaml",
                valid + "\nplatform:\n  run_id: infra-owned\n"),
            "platform control identity config was silently ignored");

    AIServerConfig dynamic_task;
    ::setenv("RL_AISERVER_MAX_AGENTS", "6", 1);
    ::setenv("RL_TASK_AGENT_COUNT", "2", 1);
    ::setenv("RL_TASK_MAP_ID", "maze-alt_2", 1);
    ::setenv("RL_TASK_MAP_EXPECTED_SHA256", std::string(64, 'a').c_str(), 1);
    ::setenv("RL_TASK_SHORTEST_ACTION_STEPS", "90", 1);
    ::setenv("RL_TASK_EPISODE_MAX_STEPS", "720", 1);
    Require(LoadDocumentWithOverrides(
                root, "dynamic-task.yaml", valid,
                AIServerConfigOverrides{aiserver_mode::kTraining},
                &dynamic_task) &&
                dynamic_task.server.max_agents == 6 &&
                dynamic_task.task.agent_num == 2 &&
                dynamic_task.task.fixed_map_id == "maze-alt_2" &&
                dynamic_task.task.fixed_map_checksum_sha256 ==
                    std::string(64, 'a') &&
                dynamic_task.task.shortest_action_steps == 90 &&
                dynamic_task.task.episode_max_steps == 720 &&
                dynamic_task.task.task_config_digest.hex.size() == 64 &&
                dynamic_task.task.task_config_digest.hex !=
                    "2502369d3df20d5c02001e7481cacd6c5be32c263cb88bdb2a40db2aee4bb167",
            "valid dynamic Agent/map assignment was rejected");

    ::setenv("RL_AISERVER_MAX_AGENTS", "1", 1);
    Require(!LoadDocument(root, "capacity-too-small.yaml", valid),
            "agent_count greater than max_agents was accepted");
    ::setenv("RL_AISERVER_MAX_AGENTS", "6", 1);
    ::setenv("RL_TASK_MAP_ID", "../escape", 1);
    Require(!LoadDocument(root, "invalid-map-id.yaml", valid),
            "unsafe map id was accepted");
    ::setenv("RL_TASK_MAP_ID", "maze-alt_2", 1);
    ::setenv("RL_TASK_MAP_EXPECTED_SHA256", "ABC", 1);
    Require(!LoadDocument(root, "invalid-map-digest.yaml", valid),
            "invalid map digest was accepted");
    ::setenv("RL_TASK_MAP_EXPECTED_SHA256", std::string(64, 'a').c_str(), 1);
    ::setenv("RL_TASK_AGENT_COUNT", "", 1);
    Require(!LoadDocument(root, "empty-agent-env.yaml", valid),
            "empty Agent environment override was accepted");
    ::setenv("RL_TASK_AGENT_COUNT", "2suffix", 1);
    Require(!LoadDocument(root, "malformed-agent-env.yaml", valid),
            "malformed Agent environment override was accepted");
    ::setenv("RL_TASK_AGENT_COUNT", "2", 1);
    ::setenv("RL_TASK_MAP_DIGSET", "typo", 1);
    Require(!LoadDocument(root, "unknown-task-env.yaml", valid),
            "unknown task environment override was ignored");
    ::unsetenv("RL_TASK_MAP_DIGSET");
    ::unsetenv("RL_AISERVER_MAX_AGENTS");
    ::unsetenv("RL_TASK_AGENT_COUNT");
    ::unsetenv("RL_TASK_MAP_ID");
    ::unsetenv("RL_TASK_MAP_EXPECTED_SHA256");
    ::unsetenv("RL_TASK_SHORTEST_ACTION_STEPS");
    ::unsetenv("RL_TASK_EPISODE_MAX_STEPS");
    for (const char* name : {"RL_TASK_ID", "RL_RUN_ID",
                             "RL_POD_ATTEMPT_ID"}) {
        ::setenv(name, "infra-owned", 1);
        Require(!LoadDocument(
                    root, std::string("platform-") + name + ".yaml", valid),
                "platform control identity was accepted by AIServer config");
        ::unsetenv(name);
    }

    ::setenv("AISERVER_CONFIG", "/retired/config.yaml", 1);
    ::setenv("RL_AISERVER_RUN_MODE", "evaluation", 1);
    ::setenv("RL_LOCAL_TRAIN_ROOT", "/retired/train", 1);
    ::setenv("RL_SAMPLE_DISTRIBUTOR_HOST", "retired-sample", 1);
    ::setenv("RL_MODEL_DISTRIBUTOR_HOST", "retired-model", 1);
    ::setenv("RL_SAMPLE_FRAGMENT_SIZE", "128suffix", 1);
    Require(!LoadDocument(root, "unknown-aiserver-env.yaml", valid),
            "unknown AIServer environment alias was ignored");
    ::unsetenv("RL_AISERVER_RUN_MODE");
    Require(LoadDocument(root, "ignored-env.yaml", valid),
            "non-component legacy environment aliases must not override config");
    ::unsetenv("AISERVER_CONFIG");
    ::unsetenv("RL_LOCAL_TRAIN_ROOT");
    ::unsetenv("RL_SAMPLE_DISTRIBUTOR_HOST");
    ::unsetenv("RL_MODEL_DISTRIBUTOR_HOST");
    ::unsetenv("RL_SAMPLE_FRAGMENT_SIZE");

    std::error_code error;
    std::filesystem::remove_all(root, error);
    return 0;
}
