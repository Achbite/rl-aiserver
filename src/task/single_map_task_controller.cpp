#include "task/single_map_task_controller.h"

#include <iomanip>
#include <sstream>

namespace {

bool SameModel(const SingleMapModelIdentity& lhs,
               const SingleMapModelIdentity& rhs) {
    return lhs.model_step == rhs.model_step &&
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

bool SingleMapTaskController::ValidateTrainingProgress(
    const SingleMapModelIdentity& model,
    int64_t produced_samples,
    int64_t& trained_samples_delta,
    std::string& error) const {
    trained_samples_delta = 0;
    if (!ValidateModel(model, error)) return false;
    if (produced_samples < 0 ||
        produced_samples < latest_produced_samples_) {
        error = "single-map produced-sample counter moved backwards";
        return false;
    }
    if (model.model_step < baseline_model_.model_step ||
        model.train_updates < baseline_model_.train_updates ||
        model.trained_samples < baseline_model_.trained_samples) {
        error = "single-map model counters moved behind the resume baseline";
        return false;
    }

    if (model.model_step < latest_model_.model_step ||
        model.train_updates < latest_model_.train_updates ||
        model.trained_samples < latest_model_.trained_samples) {
        error = "single-map active model counters moved backwards";
        return false;
    }
    if (model.model_step == latest_model_.model_step &&
        !SameModel(model, latest_model_)) {
        error = "single-map active model identity changed at one step";
        return false;
    }

    trained_samples_delta =
        model.trained_samples - baseline_model_.trained_samples;
    if (trained_samples_delta > produced_samples) {
        error = "single-map trained samples exceed produced samples";
        return false;
    }
    return true;
}

bool SingleMapTaskController::Initialize(
    int episode_max_steps,
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
    if (episode_max_steps <= 0 || initial_produced_samples != 0) {
        error = "single-map training must start with zero produced samples";
        return false;
    }
    if (initial_model.model_step != 0 ||
        initial_model.train_updates != 0 ||
        initial_model.trained_samples != 0) {
        error = "single-map training must start at model step 0 with zero counters";
        return false;
    }
    episode_max_steps_ = episode_max_steps;
    baseline_model_ = initial_model;
    latest_model_ = initial_model;
    latest_produced_samples_ = 0;
    initialized_ = true;
    error.clear();
    return true;
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
    if (!ValidateTrainingProgress(active_model, produced_samples,
                                  trained_samples_delta, error)) {
        return false;
    }

    plan.continue_task = true;
    plan.max_steps = episode_max_steps_;
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
    if (!ValidateTrainingProgress(active_model, produced_samples,
                                  trained_samples_delta, error)) {
        return false;
    }
    // This method only advances the monotonic training ledger. Capacity and
    // transport own flow control; explicit external stop owns the process
    // lifetime. Model evaluation cannot pause or terminate training.
    latest_model_ = active_model;
    latest_produced_samples_ = produced_samples;
    error.clear();
    return true;
}

SingleMapTaskSnapshot SingleMapTaskController::GetSnapshot() const {
    SingleMapTaskSnapshot snapshot;
    snapshot.initialized = initialized_;
    snapshot.baseline_model_step = baseline_model_.model_step;
    snapshot.baseline_model_checksum = baseline_model_.model_checksum;
    snapshot.baseline_train_updates = baseline_model_.train_updates;
    snapshot.baseline_trained_samples = baseline_model_.trained_samples;
    snapshot.produced_samples = latest_produced_samples_;
    snapshot.episode_max_steps = episode_max_steps_;
    return snapshot;
}

std::string SingleMapTaskController::ToJson() const {
    const auto snapshot = GetSnapshot();
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\"schema_version\":3"
           << ",\"baseline\":{\"model_step\":"
           << snapshot.baseline_model_step
           << ",\"model_checksum\":\""
           << snapshot.baseline_model_checksum
           << "\",\"train_updates\":"
           << snapshot.baseline_train_updates
           << ",\"trained_samples\":"
           << snapshot.baseline_trained_samples << '}'
           << ",\"produced_samples\":"
           << snapshot.produced_samples
           << ",\"episode_max_steps\":" << snapshot.episode_max_steps
           << '}';
    return output.str();
}
