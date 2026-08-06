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
    const std::string valid = ReadFile(argv[1]);
    const auto root = std::filesystem::temp_directory_path() /
                      ("rl-aiserver-config-test-" +
                       std::to_string(::getpid()));
    std::filesystem::create_directories(root);

    Require(LoadDocument(root, "valid.yaml", valid),
            "valid config was rejected");
    Require(!LoadDocument(
                root, "bad-revision.yaml",
                ReplaceOnce(valid, "  task_revision: 1\n",
                            "  task_revision: 1suffix\n")),
            "malformed task revision did not fail closed");
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

    ::setenv("RL_SAMPLE_FRAGMENT_SIZE", "128suffix", 1);
    Require(!LoadDocument(root, "bad-env.yaml", valid),
            "malformed environment override did not fail closed");
    ::unsetenv("RL_SAMPLE_FRAGMENT_SIZE");

    std::error_code error;
    std::filesystem::remove_all(root, error);
    return 0;
}
