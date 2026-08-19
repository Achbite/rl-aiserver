#include "ai/onnx_inferencer.h"
#include "task/episode_action_policy.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    Require(argc == 2,
            "usage: onnx_actor_wire_integration_test MODEL");
    const std::string model_path = argv[1];
    const std::vector<float> observation(17, 0.0f);
    std::string error;

    OnnxInferencer inferencer;
    Require(inferencer.LoadModel(model_path, 17, 9, &error),
            "AIServer loads the ONNX model: " + error);
    std::vector<float> logits;
    float value = 0.0f;
    Require(inferencer.Infer(observation, 17, logits, value),
            "AIServer performs inference");
    Require(logits.size() == 9 && std::isfinite(value),
            "model output has the expected shape and finite values");
    Require(ValidateEpisodeModelOutput(logits, 9, value, error),
            "model output enters the action chain: " + error);

    std::mt19937 generator(0);
    int action = -1;
    float log_probability = 0.0f;
    Require(SelectEpisodeAction(
                logits, maze::EPISODE_MODE_TRAINING, 1.0,
                generator, action, log_probability, error),
            "action selector consumes the ONNX output: " + error);
    Require(action >= 0 && action < 9 &&
                std::isfinite(log_probability),
            "selected action is valid");

    OnnxInferencer::PreparedModel prepared;
    Require(inferencer.PrepareModel(
                model_path, 17, 9, prepared, &error) && prepared.valid(),
            "model session prepares before activation: " + error);
    inferencer.ActivatePreparedModel(std::move(prepared));
    Require(inferencer.Infer(observation, 17, logits, value),
            "activated model session continues inference");
    return 0;
}
