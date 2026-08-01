#include "config/run_mode.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << std::endl;
        std::exit(1);
    }
}

}  // namespace

int main() {
    Require(aiserver_mode::Parse("1") == aiserver_mode::kTraining,
            "numeric train mode");
    Require(aiserver_mode::Parse("train") == aiserver_mode::kTraining,
            "train alias");
    Require(aiserver_mode::Parse("training") ==
                aiserver_mode::kTraining,
            "training workload");
    Require(aiserver_mode::Parse("sample-flow") == 0,
            "removed sample-flow workload");
    Require(aiserver_mode::Parse("2") == aiserver_mode::kLocalTest,
            "numeric local-test mode");
    Require(aiserver_mode::Parse("local-test") ==
                aiserver_mode::kLocalTest,
            "local-test alias");
    Require(aiserver_mode::Parse("inference-smoke") ==
                aiserver_mode::kLocalTest,
            "inference-smoke workload");
    Require(aiserver_mode::Parse("3") ==
                aiserver_mode::kModelEvaluation,
            "model-evaluation mode");
    Require(aiserver_mode::Parse("4") == aiserver_mode::kAstarTest,
            "A* mode");
    Require(aiserver_mode::Parse("invalid") == 0,
            "invalid mode");
    Require(std::string(aiserver_mode::Workload(
                aiserver_mode::kLocalTest)) == "local-test",
            "local-test canonical workload");
    Require(std::string(aiserver_mode::Workload(
                aiserver_mode::kTraining)) == "training",
            "train canonical workload");
    return 0;
}
