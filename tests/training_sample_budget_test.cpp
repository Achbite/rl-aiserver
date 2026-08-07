#include "task/training_sample_budget.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    std::string error;
    TrainingSampleBudget disabled(0, 512);
    Require(disabled.Validate(error), "zero disables the budget");
    Require(!disabled.enabled(), "zero budget must be disabled");

    TrainingSampleBudget closure(3072, 512);
    Require(closure.Validate(error), "3072/512 must be valid");
    Require(closure.effective_samples() == 3072,
            "closure budget must remain exact");
    Require(!closure.reached(2560), "budget must not stop early");
    Require(closure.reached(3072), "budget must stop at boundary");
    Require(closure.exceeded(3584), "overrun must be detectable");

    TrainingSampleBudget course_cap(1000000, 512);
    Require(course_cap.Validate(error), "course cap must be valid");
    Require(course_cap.effective_samples() == 999936,
            "a maximum must round down to a trainable sample quantum");
    Require(course_cap.effective_samples() <=
                course_cap.requested_samples(),
            "effective cap must never exceed the requested maximum");

    TrainingSampleBudget too_small(128, 512);
    Require(!too_small.Validate(error),
            "a partial sample quantum budget must fail closed");
    return 0;
}
