#include "task/single_map_task_controller.h"

#include <iomanip>
#include <limits>
#include <sstream>

namespace {

bool SameModel(const SingleMapModelIdentity& lhs,
               const SingleMapModelIdentity& rhs) {
    return lhs.model_version == rhs.model_version &&
           lhs.model_checksum == rhs.model_checksum &&
           lhs.train_updates == rhs.train_updates &&
           lhs.trained_samples == rhs.trained_samples;
}

}  // namespace

bool SingleMapTaskController::ValidateModel(
    const SingleMapModelIdentity& model,
    std::string& error) const {
    if (model.model_checksum.size() != 64 ||
        model.train_updates < 0 || model.trained_samples < 0) {
        error = "single-map model identity is invalid";
        return false;
    }
    return true;
}

bool SingleMapTaskController::ValidateRunProgress(
    const SingleMapModelIdentity& model,
    int64_t produced_samples,
    int64_t& trained_samples_delta,
    std::string& error) const {
    trained_samples_delta = 0;
    if (!ValidateModel(model, error)) return false;
    if (produced_samples < 0 ||
        produced_samples < latest_produced_samples_) {
        error = "single-map run-produced counter moved backwards";
        return false;
    }
    if (model.model_version < baseline_model_.model_version ||
        model.train_updates < baseline_model_.train_updates ||
        model.trained_samples < baseline_model_.trained_samples) {
        error = "single-map model counters moved behind the resume baseline";
        return false;
    }

    if (model.model_version < latest_model_.model_version ||
        model.train_updates < latest_model_.train_updates ||
        model.trained_samples < latest_model_.trained_samples) {
        error = "single-map active model counters moved backwards";
        return false;
    }
    if (model.model_version == latest_model_.model_version &&
        !SameModel(model, latest_model_)) {
        error = "single-map active model identity changed at one version";
        return false;
    }

    trained_samples_delta =
        model.trained_samples - baseline_model_.trained_samples;
    if (trained_samples_delta > produced_samples) {
        error = "single-map trained samples exceed run-produced samples";
        return false;
    }
    return true;
}

bool SingleMapTaskController::Initialize(
    int shortest_action_steps,
    const SingleMapModelIdentity& initial_model,
    int64_t initial_produced_samples,
    std::string& error) {
    if (initialized_) {
        error = "single-map task controller is already initialized";
        return false;
    }
    if (!ValidateModel(initial_model, error)) {
        return false;
    }
    if (shortest_action_steps <= 0 || initial_produced_samples != 0) {
        error = "single-map task run must start with zero produced samples";
        return false;
    }
    const bool fresh =
        initial_model.model_version == 0 &&
        initial_model.train_updates == 0 &&
        initial_model.trained_samples == 0;
    shortest_action_steps_ = shortest_action_steps;
    startup_mode_ = fresh ? SingleMapTaskStartupMode::Fresh
                          : SingleMapTaskStartupMode::Resume;
    baseline_model_ = initial_model;
    latest_model_ = initial_model;
    stage_ = maze::CURRICULUM_STAGE_8X;
    latest_produced_samples_ = 0;
    initialized_ = true;
    error.clear();
    return true;
}

int SingleMapTaskController::StageMultiplier() const {
    switch (stage_) {
        case maze::CURRICULUM_STAGE_8X:
            return 8;
        case maze::CURRICULUM_STAGE_4X:
            return 4;
        case maze::CURRICULUM_STAGE_2X:
            return 2;
        default:
            return 0;
    }
}

bool SingleMapTaskController::PlanNextEpisode(
    const SingleMapModelIdentity& active_model,
    int64_t produced_samples,
    SingleMapEpisodePlan& plan,
    std::string& error) {
    plan = SingleMapEpisodePlan{};
    if (!initialized_) {
        error = "single-map task controller is not initialized";
        return false;
    }
    int64_t trained_samples_delta = 0;
    if (!ValidateRunProgress(active_model, produced_samples,
                             trained_samples_delta, error)) {
        return false;
    }

    if (stage_ == maze::CURRICULUM_STAGE_COMPLETE ||
        stage_ == maze::CURRICULUM_STAGE_FAILED) {
        plan.continue_task = false;
        plan.curriculum_stage = stage_;
        plan.episode_mode = maze::EPISODE_MODE_UNSPECIFIED;
        plan.model = active_model;
        latest_model_ = active_model;
        latest_produced_samples_ = produced_samples;
        return true;
    }

    const int multiplier = StageMultiplier();
    if (multiplier <= 0 ||
        shortest_action_steps_ >
            std::numeric_limits<int>::max() / multiplier) {
        error = "single-map episode horizon would overflow";
        return false;
    }

    plan.continue_task = true;
    plan.curriculum_stage = stage_;
    plan.max_steps = shortest_action_steps_ * multiplier;
    plan.episode_mode = maze::EPISODE_MODE_TRAINING;
    plan.model = active_model;
    latest_model_ = active_model;
    latest_produced_samples_ = produced_samples;
    error.clear();
    return true;
}

bool SingleMapTaskController::ObserveTrainingProgress(
    const SingleMapModelIdentity& active_model,
    int64_t produced_samples,
    std::string& error) {
    if (!initialized_) {
        error = "single-map task controller is not initialized";
        return false;
    }
    int64_t trained_samples_delta = 0;
    if (!ValidateRunProgress(active_model, produced_samples,
                             trained_samples_delta, error)) {
        return false;
    }
    // This method only advances the monotonic training ledger. Capacity and
    // transport own flow control; explicit external stop owns run lifetime.
    // Model evaluation cannot pause or terminate a training run.
    latest_model_ = active_model;
    latest_produced_samples_ = produced_samples;
    error.clear();
    return true;
}

bool SingleMapTaskController::ReconcileDiscardedTrainingSamples(
    int64_t sample_count,
    std::string& error) {
    if (!initialized_ || sample_count < 0) {
        error = "single-map discarded sample disposition is invalid";
        return false;
    }
    if (sample_count == 0) return true;
    if (stage_ == maze::CURRICULUM_STAGE_COMPLETE ||
        stage_ == maze::CURRICULUM_STAGE_FAILED) {
        error = "single-map samples cannot be discarded outside training";
        return false;
    }
    if (sample_count > latest_produced_samples_) {
        error = "single-map discarded samples exceed the current run";
        return false;
    }

    // Producer freshness is decided asynchronously by the Sample Pool. A
    // rejected fragment was tentatively counted when its transitions were
    // created, so rewind the current-stage ledger before evaluating the next
    // monotonic counter snapshot before the next training episode.
    latest_produced_samples_ -= sample_count;
    return true;
}

SingleMapTaskSnapshot SingleMapTaskController::GetSnapshot() const {
    SingleMapTaskSnapshot snapshot;
    snapshot.initialized = initialized_;
    snapshot.startup_mode = startup_mode_;
    snapshot.baseline_model_version = baseline_model_.model_version;
    snapshot.baseline_model_checksum = baseline_model_.model_checksum;
    snapshot.baseline_train_updates = baseline_model_.train_updates;
    snapshot.baseline_trained_samples = baseline_model_.trained_samples;
    snapshot.run_produced_samples = latest_produced_samples_;
    snapshot.curriculum_stage = stage_;
    snapshot.complete = stage_ == maze::CURRICULUM_STAGE_COMPLETE;
    snapshot.failed = stage_ == maze::CURRICULUM_STAGE_FAILED;
    return snapshot;
}

const char* SingleMapTaskController::StageName(
    maze::CurriculumStage stage) {
    switch (stage) {
        case maze::CURRICULUM_STAGE_8X: return "8x";
        case maze::CURRICULUM_STAGE_4X: return "4x";
        case maze::CURRICULUM_STAGE_2X: return "2x";
        case maze::CURRICULUM_STAGE_COMPLETE: return "complete";
        case maze::CURRICULUM_STAGE_FAILED: return "failed";
        default: return "unspecified";
    }
}

const char* SingleMapTaskController::StartupModeName(
    SingleMapTaskStartupMode mode) {
    switch (mode) {
        case SingleMapTaskStartupMode::Fresh: return "fresh";
        case SingleMapTaskStartupMode::Resume: return "resume";
        default: return "unspecified";
    }
}

std::string SingleMapTaskController::ToJson() const {
    const auto snapshot = GetSnapshot();
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\"schema_version\":2"
           << ",\"startup_mode\":\""
           << StartupModeName(snapshot.startup_mode) << "\""
           << ",\"baseline\":{\"model_version\":"
           << snapshot.baseline_model_version
           << ",\"model_checksum\":\""
           << snapshot.baseline_model_checksum
           << "\",\"train_updates\":"
           << snapshot.baseline_train_updates
           << ",\"trained_samples\":"
           << snapshot.baseline_trained_samples << '}'
           << ",\"run_produced_samples\":"
           << snapshot.run_produced_samples
           << ",\"episode_horizon_multiplier\":" << StageMultiplier()
           << ",\"complete\":" << (snapshot.complete ? "true" : "false")
           << ",\"failed\":" << (snapshot.failed ? "true" : "false")
           << '}';
    return output.str();
}
