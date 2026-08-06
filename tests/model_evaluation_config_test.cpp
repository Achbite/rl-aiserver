#include "config/config_loader.h"

#include <iostream>
#include <string>

int main() {
    ModelConfig config;
    config.evaluation_dir = "models/evaluation/000200/";
    const std::string expected =
        "models/evaluation/000200/SaveModel.onnx";
    const std::string actual = LocalEvaluationModelPath(config);
    if (actual != expected) {
        std::cerr << "unexpected evaluation model path: " << actual << "\n";
        return 1;
    }
    const std::string expected_manifest =
        "models/evaluation/000200/manifest.json";
    const std::string actual_manifest =
        LocalEvaluationManifestPath(config);
    if (actual_manifest != expected_manifest) {
        std::cerr << "unexpected evaluation manifest path: "
                  << actual_manifest << "\n";
        return 1;
    }
    return 0;
}
