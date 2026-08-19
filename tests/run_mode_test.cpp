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
    Require(aiserver_mode::Parse("training") ==
                aiserver_mode::kTraining,
            "training workload");
    Require(aiserver_mode::Parse("evaluation") ==
                aiserver_mode::kEvaluation,
            "evaluation workload");
    Require(std::string(aiserver_mode::Workload(
                aiserver_mode::kEvaluation)) == "evaluation",
            "evaluation canonical workload");
    Require(std::string(aiserver_mode::Workload(
                aiserver_mode::kTraining)) == "training",
            "train canonical workload");
    Require(aiserver_mode::ExposesTrainingStatus(
                aiserver_mode::kTraining),
            "training exposes the Learner status service");
    Require(!aiserver_mode::ExposesTrainingStatus(
                aiserver_mode::kEvaluation),
            "evaluation must not impersonate a training Actor");
    for (const std::string& retired : {
             "1", "2", "3", "4", "train", "local-test",
             "inference-smoke", "model-evaluation", "map-validation",
         }) {
        Require(aiserver_mode::Parse(retired) == 0,
                "retired workload must fail closed: " + retired);
    }
    return 0;
}
