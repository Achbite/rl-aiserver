#include "task/single_map_task_controller.h"

#include <cassert>
#include <iostream>
#include <limits>
#include <string>

namespace {

SingleMapModelIdentity Model(ModelVersion version,
                             int64_t train_updates,
                             int64_t trained_samples) {
    SingleMapModelIdentity model;
    model.model_version = version;
    model.model_checksum =
        std::string(64, static_cast<char>('a' + version % 20));
    model.train_updates = train_updates;
    model.trained_samples = trained_samples;
    return model;
}

SingleMapModelIdentity Model(ModelVersion version, int64_t trained_samples) {
    return Model(version, version, trained_samples);
}

}  // namespace

int main() {
    std::string error;
    SingleMapTaskController controller;
    assert(controller.Initialize(188, Model(0, 0), 0, error));
    assert(controller.GetSnapshot().startup_mode ==
           SingleMapTaskStartupMode::Fresh);

    SingleMapEpisodePlan plan;
    assert(controller.PlanNextEpisode(Model(0, 0), 0, plan, error));
    assert(plan.continue_task);
    assert(plan.episode_mode == maze::EPISODE_MODE_TRAINING);
    assert(plan.curriculum_stage == maze::CURRICULUM_STAGE_8X);
    assert(plan.max_steps == 1504);

    // Model-quality evaluation is external to training. Crossing the former
    // 100k/200k thresholds and the retired one-million-sample boundary must
    // not pause, complete, or change the episode mode.
    const struct {
        ModelVersion model_version;
        int64_t samples;
    } progress[] = {
        {195, 99840},
        {196, 100352},
        {390, 200192},
        {1953, 999936},
        {1954, 1000448},
        {3907, 2000384},
    };
    for (const auto& item : progress) {
        assert(controller.ObserveTrainingProgress(
            Model(item.model_version, item.samples), item.samples, error));
        assert(controller.PlanNextEpisode(
            Model(item.model_version, item.samples), item.samples,
            plan, error));
        assert(plan.continue_task);
        assert(plan.episode_mode == maze::EPISODE_MODE_TRAINING);
    }
    const auto training_snapshot = controller.GetSnapshot();
    const std::string training_receipt = controller.ToJson();
    assert(training_receipt.find("evaluation") == std::string::npos);
    assert(training_snapshot.run_produced_samples == 2000384);

    // Sample Pool freshness reconciliation remains part of the training
    // ledger and cannot move the run-produced counter below zero.
    SingleMapTaskController freshness;
    error.clear();
    assert(freshness.Initialize(188, Model(0, 0), 0, error));
    assert(freshness.ObserveTrainingProgress(
        Model(1, 512), 640, error));
    assert(freshness.ReconcileDiscardedTrainingSamples(128, error));
    assert(freshness.GetSnapshot().run_produced_samples == 512);
    assert(!freshness.ReconcileDiscardedTrainingSamples(513, error));

    // Resume starts a new run ledger at zero while preserving publication and
    // optimizer counters as an immutable baseline. It never schedules an
    // evaluation when the relative trained count crosses an old threshold.
    SingleMapTaskController resumed;
    const auto resume_model = Model(201, 200, 10000);
    error.clear();
    assert(resumed.Initialize(188, resume_model, 0, error));
    auto resume_snapshot = resumed.GetSnapshot();
    assert(resume_snapshot.startup_mode ==
           SingleMapTaskStartupMode::Resume);
    assert(resume_snapshot.baseline_model_version == 201);
    assert(resume_snapshot.baseline_train_updates == 200);
    assert(resume_snapshot.baseline_trained_samples == 10000);
    assert(resume_snapshot.run_produced_samples == 0);
    assert(resumed.ObserveTrainingProgress(
        Model(202, 200, 10000), 0, error));
    assert(resumed.ObserveTrainingProgress(
        Model(203, 201, 10050), 50, error));
    assert(resumed.ObserveTrainingProgress(
        Model(204, 202, 10100), 100, error));
    assert(resumed.PlanNextEpisode(
        Model(204, 202, 10100), 100, plan, error));
    assert(plan.episode_mode == maze::EPISODE_MODE_TRAINING);

    // Publication identity uses the full uint64 wire domain. Crossing the
    // former signed-int boundary is ordinary progress; UINT64_MAX cannot
    // wrap to zero.
    SingleMapTaskController high_version;
    const ModelVersion above_int_max =
        static_cast<ModelVersion>(std::numeric_limits<int>::max()) + 1;
    error.clear();
    assert(high_version.Initialize(
        188, Model(above_int_max, 0, 0), 0, error));
    assert(high_version.ObserveTrainingProgress(
        Model(std::numeric_limits<ModelVersion>::max(), 1, 0), 0, error));
    const std::string before_wrap = high_version.ToJson();
    assert(!high_version.ObserveTrainingProgress(Model(0, 2, 0), 0, error));
    assert(high_version.ToJson() == before_wrap);

    // Publication and optimizer counters advance independently, but neither
    // training counter may roll back. Rejected observations are atomic.
    const std::string before_invalid_progress = resumed.ToJson();
    assert(!resumed.ObserveTrainingProgress(
        Model(205, 201, 10100), 100, error));
    assert(resumed.ToJson() == before_invalid_progress);
    error.clear();
    assert(!resumed.ObserveTrainingProgress(
        Model(205, 203, 10099), 100, error));
    assert(resumed.ToJson() == before_invalid_progress);
    error.clear();
    assert(!resumed.ObserveTrainingProgress(
        Model(203, 203, 10100), 100, error));
    assert(resumed.ToJson() == before_invalid_progress);
    error.clear();
    assert(!resumed.ObserveTrainingProgress(
        Model(204, 203, 10100), 100, error));
    assert(resumed.ToJson() == before_invalid_progress);
    error.clear();
    assert(!resumed.ObserveTrainingProgress(
        Model(205, 203, 10101), 100, error));
    assert(resumed.ToJson() == before_invalid_progress);

    SingleMapTaskController invalid_resume;
    error.clear();
    assert(!invalid_resume.Initialize(
        188, Model(1, -1, 0), 0, error));
    assert(!invalid_resume.GetSnapshot().initialized);

    // Removing the evaluation counter also removes its near-uint64 overflow
    // failure. Horizon overflow remains fail-closed and atomic.
    SingleMapTaskController high_counter;
    const int64_t near_max = std::numeric_limits<int64_t>::max() - 1;
    error.clear();
    assert(high_counter.Initialize(
        188, Model(1, 0, near_max), 0, error));
    assert(high_counter.PlanNextEpisode(
        Model(2, 1, near_max), 0, plan, error));
    assert(plan.episode_mode == maze::EPISODE_MODE_TRAINING);

    SingleMapTaskController horizon_overflow;
    error.clear();
    assert(horizon_overflow.Initialize(
        std::numeric_limits<int>::max(), Model(0, 0), 0, error));
    const std::string before_overflow = horizon_overflow.ToJson();
    assert(!horizon_overflow.PlanNextEpisode(
        Model(0, 0), 0, plan, error));
    assert(horizon_overflow.ToJson() == before_overflow);

    std::cout << "single-map training controller contract passed\n";
    return 0;
}
