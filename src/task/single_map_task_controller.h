#pragma once

#include "contracts/contract_namespaces.h"

#include <cstdint>
#include <string>
#include <vector>

struct SingleMapModelIdentity {
    int model_version = -1;
    std::string model_checksum;
    int64_t trained_samples = 0;
};

struct SingleMapAgentEvaluation {
    bool success = false;
    int64_t transition_count = 0;
};

struct SingleMapEpisodePlan {
    bool continue_task = true;
    maze::EpisodeMode episode_mode = maze::EPISODE_MODE_TRAINING;
    maze::CurriculumStage curriculum_stage =
        maze::CURRICULUM_STAGE_8X;
    int max_steps = 0;
    SingleMapModelIdentity model;
};

struct SingleMapTaskControllerConfig {
    int agent_num = 4;
    int64_t sample_quantum = 512;
    int64_t stage_8x_sample_budget = 1000000;
    int64_t stage_4x_sample_budget = 500000;
    int64_t stage_2x_sample_budget = 500000;
    int64_t evaluation_interval_samples = 100000;
    int evaluation_episodes_per_round = 25;
    double stage_8x_success_threshold = 0.80;
    double stage_4x_success_threshold = 0.90;
    double stage_2x_success_threshold = 0.90;
    double final_path_ratio_median_limit = 1.50;
    double final_path_ratio_p95_limit = 2.00;
};

struct SingleMapEvaluationRecord {
    maze::CurriculumStage curriculum_stage =
        maze::CURRICULUM_STAGE_UNSPECIFIED;
    SingleMapModelIdentity model;
    bool final_attempt = false;
    int64_t stage_produced_samples = 0;
    int64_t stage_sample_budget = 0;
    double argmax_round_1_success_rate = 0.0;
    double argmax_round_2_success_rate = 0.0;
    double stochastic_success_rate = 0.0;
    double successful_path_ratio_median = 0.0;
    double successful_path_ratio_p95 = 0.0;
    bool passed = false;
};

struct SingleMapTaskSnapshot {
    bool initialized = false;
    bool evaluation_active = false;
    maze::CurriculumStage curriculum_stage =
        maze::CURRICULUM_STAGE_UNSPECIFIED;
    maze::EpisodeMode evaluation_mode =
        maze::EPISODE_MODE_UNSPECIFIED;
    int evaluation_round = 0;
    int evaluation_episode_in_round = 0;
    int evaluation_episodes_per_round = 0;
    int evaluation_model_version = -1;
    int64_t evaluation_model_trained_samples = 0;
    int64_t stage_produced_samples = 0;
    int64_t stage_sample_budget = 0;
    int64_t next_evaluation_trained_samples = 0;
    double latest_argmax_round_1_success_rate = 0.0;
    double latest_argmax_round_2_success_rate = 0.0;
    double latest_stochastic_success_rate = 0.0;
    double latest_path_ratio_median = 0.0;
    double latest_path_ratio_p95 = 0.0;
    bool complete = false;
    bool failed = false;
};

// Owns the fixed-map curriculum and evaluation schedule. The controller sees
// only model-training counters and Agent episode outcomes; it never exposes
// map or curriculum fields to the Learner.
class SingleMapTaskController {
public:
    explicit SingleMapTaskController(
        SingleMapTaskControllerConfig config = {});

    bool Initialize(int shortest_action_steps,
                    const SingleMapModelIdentity& initial_model,
                    int64_t initial_produced_samples,
                    std::string& error);

    bool PlanNextEpisode(const SingleMapModelIdentity& active_model,
                         int64_t produced_samples,
                         SingleMapEpisodePlan& plan,
                         std::string& error);

    bool ShouldPauseTrainingCollection(
        const SingleMapModelIdentity& active_model,
        int64_t produced_samples,
        bool& should_pause,
        std::string& error);

    bool RecordEvaluationEpisode(
        maze::EpisodeMode episode_mode,
        const SingleMapModelIdentity& model,
        const std::vector<SingleMapAgentEvaluation>& agents,
        std::string& error);

    bool IsEvaluationActive() const;
    SingleMapTaskSnapshot GetSnapshot() const;
    const std::vector<SingleMapEvaluationRecord>& history() const;
    std::string ToJson() const;

private:
    struct EvaluationRound {
        maze::EpisodeMode mode = maze::EPISODE_MODE_UNSPECIFIED;
        int environment_episodes = 0;
        int64_t agent_episodes = 0;
        int64_t successes = 0;
        std::vector<double> successful_path_ratios;
    };

    struct EvaluationCampaign {
        bool active = false;
        bool final_attempt = false;
        SingleMapModelIdentity model;
        int64_t produced_samples = 0;
        int phase = 0;
        EvaluationRound rounds[3];
    };

    bool ValidateConfig(std::string& error) const;
    bool ValidateModel(const SingleMapModelIdentity& model,
                       std::string& error) const;
    int StageMultiplier() const;
    int64_t StageBudget() const;
    int64_t StageEffectiveBudget() const;
    double StageSuccessThreshold() const;
    void StartEvaluation(const SingleMapModelIdentity& model,
                         int64_t produced_samples,
                         bool final_attempt);
    void FinishEvaluation();
    static double SuccessRate(const EvaluationRound& round);
    static double Quantile(std::vector<double> values, double quantile);
    static const char* StageName(maze::CurriculumStage stage);
    static const char* ModeName(maze::EpisodeMode mode);

    SingleMapTaskControllerConfig config_;
    bool initialized_ = false;
    int shortest_action_steps_ = 0;
    maze::CurriculumStage stage_ = maze::CURRICULUM_STAGE_UNSPECIFIED;
    int64_t stage_start_produced_samples_ = 0;
    int64_t latest_produced_samples_ = 0;
    int64_t next_evaluation_trained_samples_ = 0;
    EvaluationCampaign evaluation_;
    std::vector<SingleMapEvaluationRecord> history_;
};
