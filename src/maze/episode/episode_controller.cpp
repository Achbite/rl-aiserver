#include "maze/episode/episode_controller.h"

#include <iomanip>
#include <sstream>

namespace {

bool SameModel(const MazeModelIdentity& lhs,
               const MazeModelIdentity& rhs) {
    return lhs.model_step == rhs.model_step &&
           lhs.model_lineage_id == rhs.model_lineage_id &&
           lhs.trained_samples == rhs.trained_samples;
}

}  // namespace

bool MazeEpisodeController::ValidateModel(
    const MazeModelIdentity& model,
    std::string& error) const {
    if (model.model_lineage_id.empty() || model.trained_samples < 0) {
        error = "single-map model identity is invalid";
        return false;
    }
    return true;
}

bool MazeEpisodeController::ValidateTrainingProgress(
    const MazeModelIdentity& model,
    int64_t produced_transitions,
    std::string& error) const {
    if (!ValidateModel(model, error)) return false;
    if (produced_transitions < 0 ||
        produced_transitions < latest_produced_transitions_) {
        error = "single-map produced-transition counter moved backwards";
        return false;
    }
    if (model.model_step < baseline_model_.model_step ||
        model.trained_samples < baseline_model_.trained_samples) {
        error = "single-map model counters moved behind the resume baseline";
        return false;
    }

    if (model.model_step < latest_model_.model_step ||
        model.trained_samples < latest_model_.trained_samples) {
        error = "single-map active model counters moved backwards";
        return false;
    }
    if (model.model_step == latest_model_.model_step &&
        !SameModel(model, latest_model_)) {
        error = "single-map active model identity changed at one step";
        return false;
    }

    // model.trained_samples is Learner-global, while produced_transitions is
    // local to this AIServer. With multiple ServerPods, the global counter can
    // legitimately exceed any one producer's local counter. Keep both ledgers
    // monotonic here; task-wide accounting belongs to the Learner/Infra
    // aggregation boundary, where all producers are visible.
    return true;
}

bool MazeEpisodeController::Initialize(
    int episode_max_steps,
    const MazeModelIdentity& initial_model,
    int64_t initial_produced_transitions,
    std::string& error) {
    if (initialized_) {
        error = "single-map task controller is already initialized";
        return false;
    }
    if (!ValidateModel(initial_model, error)) {
        return false;
    }
    if (episode_max_steps <= 0 || initial_produced_transitions != 0) {
        error = "single-map training must start with zero produced transitions";
        return false;
    }
    if (initial_model.model_step != 0 || initial_model.trained_samples != 0) {
        error = "single-map training must start at model step 0 with zero counters";
        return false;
    }
    episode_max_steps_ = episode_max_steps;
    baseline_model_ = initial_model;
    latest_model_ = initial_model;
    latest_produced_transitions_ = 0;
    initialized_ = true;
    error.clear();
    return true;
}

bool MazeEpisodeController::PlanNextEpisode(
    const MazeModelIdentity& active_model,
    int64_t produced_transitions,
    MazeEpisodePlan& plan,
    std::string& error) {
    plan = MazeEpisodePlan{};
    if (!initialized_) {
        error = "single-map task controller is not initialized";
        return false;
    }
    if (!ValidateTrainingProgress(active_model, produced_transitions, error)) {
        return false;
    }

    plan.max_steps = episode_max_steps_;
    plan.episode_mode = maze::EPISODE_MODE_TRAINING;
    plan.model = active_model;
    latest_model_ = active_model;
    latest_produced_transitions_ = produced_transitions;
    error.clear();
    return true;
}

bool MazeEpisodeController::ObserveTrainingProgress(
    const MazeModelIdentity& active_model,
    int64_t produced_transitions,
    std::string& error) {
    if (!initialized_) {
        error = "single-map task controller is not initialized";
        return false;
    }
    if (!ValidateTrainingProgress(active_model, produced_transitions, error)) {
        return false;
    }
    // This method only advances the monotonic training ledger. Capacity and
    // transport own flow control; explicit external stop owns the process
    // lifetime. Model evaluation cannot pause or terminate training.
    latest_model_ = active_model;
    latest_produced_transitions_ = produced_transitions;
    error.clear();
    return true;
}

MazeTaskSnapshot MazeEpisodeController::GetSnapshot() const {
    MazeTaskSnapshot snapshot;
    snapshot.initialized = initialized_;
    snapshot.baseline_model_step = baseline_model_.model_step;
    snapshot.baseline_model_lineage_id = baseline_model_.model_lineage_id;
    snapshot.baseline_trained_samples = baseline_model_.trained_samples;
    snapshot.produced_transitions = latest_produced_transitions_;
    snapshot.episode_max_steps = episode_max_steps_;
    return snapshot;
}

std::string MazeEpisodeController::ToJson() const {
    const auto snapshot = GetSnapshot();
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\"baseline\":{\"model_step\":"
           << snapshot.baseline_model_step
           << ",\"model_lineage_id\":\""
           << snapshot.baseline_model_lineage_id
           << "\",\"trained_samples\":"
           << snapshot.baseline_trained_samples << '}'
           << ",\"produced_transitions\":"
           << snapshot.produced_transitions
           << ",\"episode_max_steps\":" << snapshot.episode_max_steps
           << '}';
    return output.str();
}
