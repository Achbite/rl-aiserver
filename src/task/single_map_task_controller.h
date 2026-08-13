#pragma once

#include "contracts/contract_namespaces.h"
#include "model/model_version.h"

#include <cstdint>
#include <string>

struct SingleMapModelIdentity {
    ModelVersion model_version = 0;
    std::string model_checksum;
    int64_t train_updates = 0;
    int64_t trained_samples = 0;
};

enum class SingleMapTaskStartupMode {
    Unspecified,
    Fresh,
    Resume,
};

struct SingleMapEpisodePlan {
    bool continue_task = true;
    maze::EpisodeMode episode_mode = maze::EPISODE_MODE_TRAINING;
    maze::CurriculumStage curriculum_stage =
        maze::CURRICULUM_STAGE_8X;
    int max_steps = 0;
    SingleMapModelIdentity model;
};

struct SingleMapTaskSnapshot {
    bool initialized = false;
    SingleMapTaskStartupMode startup_mode =
        SingleMapTaskStartupMode::Unspecified;
    ModelVersion baseline_model_version = 0;
    std::string baseline_model_checksum;
    int64_t baseline_train_updates = 0;
    int64_t baseline_trained_samples = 0;
    int64_t run_produced_samples = 0;
    maze::CurriculumStage curriculum_stage =
        maze::CURRICULUM_STAGE_UNSPECIFIED;
    bool complete = false;
    bool failed = false;
};

// Owns the fixed-map training ledger. Evaluation is a developer-triggered
// standalone workflow and must never pause or mutate this training run.
class SingleMapTaskController {
public:
    SingleMapTaskController() = default;

    bool Initialize(int shortest_action_steps,
                    const SingleMapModelIdentity& initial_model,
                    int64_t initial_produced_samples,
                    std::string& error);

    bool PlanNextEpisode(const SingleMapModelIdentity& active_model,
                         int64_t produced_samples,
                         SingleMapEpisodePlan& plan,
                         std::string& error);

    bool ObserveTrainingProgress(
        const SingleMapModelIdentity& active_model,
        int64_t produced_samples,
        std::string& error);

    bool ReconcileDiscardedTrainingSamples(int64_t sample_count,
                                           std::string& error);

    SingleMapTaskSnapshot GetSnapshot() const;
    std::string ToJson() const;

private:
    bool ValidateModel(const SingleMapModelIdentity& model,
                       std::string& error) const;
    bool ValidateRunProgress(const SingleMapModelIdentity& model,
                             int64_t produced_samples,
                             int64_t& trained_samples_delta,
                             std::string& error) const;
    int StageMultiplier() const;
    static const char* StageName(maze::CurriculumStage stage);
    static const char* StartupModeName(SingleMapTaskStartupMode mode);

    bool initialized_ = false;
    SingleMapTaskStartupMode startup_mode_ =
        SingleMapTaskStartupMode::Unspecified;
    SingleMapModelIdentity baseline_model_;
    SingleMapModelIdentity latest_model_;
    int shortest_action_steps_ = 0;
    maze::CurriculumStage stage_ = maze::CURRICULUM_STAGE_UNSPECIFIED;
    int64_t latest_produced_samples_ = 0;
};
