#pragma once

#include "contracts/contract_namespaces.h"
#include "model/model_step.h"

#include <cstdint>
#include <string>

struct SingleMapModelIdentity {
    ModelStep model_step = 0;
    std::string model_checksum;
    int64_t train_updates = 0;
    int64_t trained_samples = 0;
};

struct SingleMapEpisodePlan {
    bool continue_task = true;
    maze::EpisodeMode episode_mode = maze::EPISODE_MODE_TRAINING;
    int max_steps = 0;
    SingleMapModelIdentity model;
};

struct SingleMapTaskSnapshot {
    bool initialized = false;
    ModelStep baseline_model_step = 0;
    std::string baseline_model_checksum;
    int64_t baseline_train_updates = 0;
    int64_t baseline_trained_samples = 0;
    int64_t produced_transitions = 0;
    int episode_max_steps = 0;
};

// Owns the fixed-map training ledger. Evaluation is a developer-triggered
// standalone workflow and must never pause or mutate training progress.
class SingleMapTaskController {
public:
    SingleMapTaskController() = default;

    bool Initialize(int episode_max_steps,
                    const SingleMapModelIdentity& initial_model,
                    int64_t initial_produced_transitions,
                    std::string& error);

    bool PlanNextEpisode(const SingleMapModelIdentity& active_model,
                         int64_t produced_transitions,
                         SingleMapEpisodePlan& plan,
                         std::string& error);

    bool ObserveTrainingProgress(
        const SingleMapModelIdentity& active_model,
        int64_t produced_transitions,
        std::string& error);

    SingleMapTaskSnapshot GetSnapshot() const;
    std::string ToJson() const;

private:
    bool ValidateModel(const SingleMapModelIdentity& model,
                       std::string& error) const;
    bool ValidateTrainingProgress(const SingleMapModelIdentity& model,
                                  int64_t produced_transitions,
                                  int64_t& trained_samples_delta,
                                  std::string& error) const;
    bool initialized_ = false;
    SingleMapModelIdentity baseline_model_;
    SingleMapModelIdentity latest_model_;
    int episode_max_steps_ = 0;
    int64_t latest_produced_transitions_ = 0;
};
