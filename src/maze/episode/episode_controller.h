#pragma once

#include "maze/protocol/contract_types.h"
#include "task/model/model_step.h"

#include <cstdint>
#include <string>

struct MazeModelIdentity {
    ModelStep model_step = 0;
    std::string model_lineage_id;
    int64_t trained_samples = 0;
};

struct MazeEpisodePlan {
    maze::EpisodeMode episode_mode = maze::EPISODE_MODE_TRAINING;
    int max_steps = 0;
    MazeModelIdentity model;
};

struct MazeTaskSnapshot {
    bool initialized = false;
    ModelStep baseline_model_step = 0;
    std::string baseline_model_lineage_id;
    int64_t baseline_trained_samples = 0;
    int64_t produced_transitions = 0;
    int episode_max_steps = 0;
};

// Owns the fixed-map training ledger. Evaluation is a developer-triggered
// standalone workflow and must never pause or mutate training progress.
class MazeEpisodeController {
public:
    MazeEpisodeController() = default;

    bool Initialize(int episode_max_steps,
                    const MazeModelIdentity& initial_model,
                    int64_t initial_produced_transitions,
                    std::string& error);

    bool PlanNextEpisode(const MazeModelIdentity& active_model,
                         int64_t produced_transitions,
                         MazeEpisodePlan& plan,
                         std::string& error);

    bool ObserveTrainingProgress(
        const MazeModelIdentity& active_model,
        int64_t produced_transitions,
        std::string& error);

    MazeTaskSnapshot GetSnapshot() const;
    std::string ToJson() const;

private:
    bool ValidateModel(const MazeModelIdentity& model,
                       std::string& error) const;
    bool ValidateTrainingProgress(const MazeModelIdentity& model,
                                  int64_t produced_transitions,
                                  std::string& error) const;
    bool initialized_ = false;
    MazeModelIdentity baseline_model_;
    MazeModelIdentity latest_model_;
    int episode_max_steps_ = 0;
    int64_t latest_produced_transitions_ = 0;
};
