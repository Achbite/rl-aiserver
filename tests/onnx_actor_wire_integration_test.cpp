#include "ai/onnx_inferencer.h"
#include "sample/training_transition_builder.h"
#include "task/episode_action_policy.h"

#include <openssl/evp.h>

#include <cmath>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {

constexpr const char* kFixtureSha256 =
    "c3b60617f304d36b89da4b1876791b711b7d3780a7348d5641bb8b7603ed96e3";

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

bool Near(float lhs, float rhs, float tolerance = 1e-6f) {
    return std::abs(lhs - rhs) <= tolerance;
}

std::string Sha256(const std::string& bytes) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_size = 0;
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    Require(context != nullptr, "SHA256 context allocated");
    Require(EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1,
            "SHA256 initialized");
    Require(EVP_DigestUpdate(context, bytes.data(), bytes.size()) == 1,
            "SHA256 updated");
    Require(EVP_DigestFinal_ex(context, digest, &digest_size) == 1,
            "SHA256 finalized");
    EVP_MD_CTX_free(context);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index) {
        output << std::setw(2) << static_cast<int>(digest[index]);
    }
    return output.str();
}

std::string Hex(const std::string& bytes) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned char value : bytes) {
        output << std::setw(2) << static_cast<int>(value);
    }
    return output.str();
}

void TestLearnerOnnxThroughActorWire(const std::string& model_path) {
    std::ifstream input(model_path, std::ios::binary);
    Require(input.good(), "tracked ONNX fixture opens");
    const std::string model_bytes{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    Require(Sha256(model_bytes) == kFixtureSha256,
            "tracked ONNX fixture digest matches provenance");

    const std::vector<float> observation = {
        0.0f, 0.6931471824645996f, 1.3862943649291992f, 0.25f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };
    const std::vector<float> expected_logits = {
        0.0f, 0.6931471824645996f, 1.3862943649291992f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };

    OnnxInferencer inferencer;
    std::string error;
    Require(inferencer.LoadModel(model_path, 17, 9, &error),
            "production OnnxInferencer loads Learner fixture: " + error);
    std::vector<float> logits;
    float value = 0.0f;
    Require(inferencer.Infer(observation, 17, logits, value),
            "production OnnxInferencer runs Learner fixture");
    Require(logits.size() == 9, "Learner fixture emits nine logits");
    for (std::size_t index = 0; index < logits.size(); ++index) {
        Require(Near(logits[index], expected_logits[index]),
                "C++ logits match fixed Learner weights");
    }
    Require(Near(value, 0.25f), "C++ value matches fixed Learner weights");
    Require(ValidateEpisodeModelOutput(logits, 9, value, error),
            "production model-output validator accepts fixture");

    std::mt19937 rng(0);
    int action = -1;
    float log_probability = 0.0f;
    for (int attempt = 0; attempt < 4096 && action != 2; ++attempt) {
        Require(SelectEpisodeAction(
                    logits, maze::EPISODE_MODE_TRAINING, 1.0,
                    rng, action, log_probability, error),
                "production selector consumes ONNX logits: " + error);
    }
    Require(action == 2, "argmax action is fixture action two");
    Require(Near(log_probability, -1.1786550283432007f),
            "actor wire log probability matches categorical golden: " +
                std::to_string(log_probability));

    SessionManager::AgentRuntime agent;
    agent.has_pending_action = true;
    agent.pending_obs = observation;
    agent.pending_action = action;
    agent.pending_action_frame_id = 0;
    agent.pending_log_prob = log_probability;
    agent.pending_value = value;
    RewardDetail reward;
    training::Sample sample;
    const std::vector<training::Sample> fragment;
    Require(BuildTrainingSample(
                agent, observation, reward, false,
                maze::MAZE_TERMINATION_REASON_ACTIVE, 17, 9,
                fragment, sample, error),
            "production wire builder consumes selector output: " + error);
    Require(
        Hex(sample.SerializeAsString()) ==
            "0a44000000001872313f1872b13f0000803e000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001244000000001872313f1872b13f0000803e0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000018022d2bde96bf350000803e4801",
        "production ONNX-selector-sample wire matches deterministic golden");
}

void TestPreparedSessionLifetimeAndConcurrentPreparation(
    const std::string& fixture_path) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
        ("onnx-prepared-lifetime-" + std::to_string(::getpid()));
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path active_path = root / "active.onnx";
    const fs::path staged_path = root / "staged.onnx";
    fs::copy_file(fixture_path, active_path);
    fs::copy_file(fixture_path, staged_path);

    OnnxInferencer inferencer;
    std::string error;
    Require(inferencer.LoadModel(active_path.string(), 17, 9, &error),
            "active session loads from cache file: " + error);
    std::atomic<bool> preparation_done{false};
    std::atomic<bool> preparation_ok{false};
    OnnxInferencer::PreparedModel staged;
    std::thread loader([&]() {
        std::string prepare_error;
        preparation_ok.store(inferencer.PrepareModel(
            staged_path.string(), 17, 9, staged, &prepare_error));
        preparation_done.store(true);
    });

    const std::vector<float> observation(17, 0.0f);
    int inference_count = 0;
    while (!preparation_done.load()) {
        std::vector<float> logits;
        float value = 0.0f;
        Require(inferencer.Infer(observation, 17, logits, value),
                "active inference remains available during PrepareModel");
        ++inference_count;
    }
    loader.join();
    Require(preparation_ok.load() && staged.valid(),
            "background PrepareModel produces a reusable shared session");

    fs::remove(active_path);
    std::vector<float> logits;
    float value = 0.0f;
    Require(inferencer.Infer(observation, 17, logits, value),
            "active shared session survives cache file eviction");
    inferencer.ActivatePreparedModel(std::move(staged));
    fs::remove(staged_path);
    Require(inferencer.Infer(observation, 17, logits, value),
            "new shared session survives its cache file eviction");
    (void)inference_count;
    fs::remove_all(root);
}

}  // namespace

int main(int argc, char** argv) {
    Require(argc == 2, "usage: onnx_actor_wire_integration_test MODEL");
    TestLearnerOnnxThroughActorWire(argv[1]);
    TestPreparedSessionLifetimeAndConcurrentPreparation(argv[1]);
    std::cout << "onnx_actor_wire_integration: PASS\n";
    return 0;
}
