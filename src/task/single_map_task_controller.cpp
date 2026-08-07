#include "task/single_map_task_controller.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {

bool SameModel(const SingleMapModelIdentity& lhs,
               const SingleMapModelIdentity& rhs) {
    return lhs.model_version == rhs.model_version &&
           lhs.model_checksum == rhs.model_checksum &&
           lhs.trained_samples == rhs.trained_samples;
}

}  // namespace

SingleMapTaskController::SingleMapTaskController(
    SingleMapTaskControllerConfig config)
    : config_(std::move(config)) {}

bool SingleMapTaskController::ValidateConfig(std::string& error) const {
    if (config_.agent_num <= 0 ||
        config_.sample_quantum <= 0 ||
        config_.stage_8x_sample_budget <= 0 ||
        config_.stage_4x_sample_budget <= 0 ||
        config_.stage_2x_sample_budget <= 0 ||
        config_.evaluation_interval_samples <= 0 ||
        config_.evaluation_episodes_per_round <= 0) {
        error = "single-map task counts and budgets must be positive";
        return false;
    }
    const int64_t budgets[] = {
        config_.stage_8x_sample_budget,
        config_.stage_4x_sample_budget,
        config_.stage_2x_sample_budget,
    };
    for (const int64_t budget : budgets) {
        if (budget - budget % config_.sample_quantum <= 0) {
            error = "single-map stage budget must contain a trainable batch";
            return false;
        }
    }
    const double thresholds[] = {
        config_.stage_8x_success_threshold,
        config_.stage_4x_success_threshold,
        config_.stage_2x_success_threshold,
    };
    for (double threshold : thresholds) {
        if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
            error = "single-map success thresholds must be in [0,1]";
            return false;
        }
    }
    if (!std::isfinite(config_.final_path_ratio_median_limit) ||
        !std::isfinite(config_.final_path_ratio_p95_limit) ||
        config_.final_path_ratio_median_limit <= 0.0 ||
        config_.final_path_ratio_p95_limit <= 0.0) {
        error = "single-map path ratio limits must be finite and positive";
        return false;
    }
    return true;
}

bool SingleMapTaskController::ValidateModel(
    const SingleMapModelIdentity& model,
    std::string& error) const {
    if (model.model_version < 0 || model.model_checksum.size() != 64 ||
        model.trained_samples < 0) {
        error = "single-map model identity is invalid";
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
    if (!ValidateConfig(error) || !ValidateModel(initial_model, error)) {
        return false;
    }
    if (shortest_action_steps <= 0 || initial_produced_samples != 0 ||
        initial_model.model_version != 0 ||
        initial_model.trained_samples != 0) {
        error = "single-map curriculum requires a fresh v0 model and zero samples";
        return false;
    }
    shortest_action_steps_ = shortest_action_steps;
    stage_ = maze::CURRICULUM_STAGE_8X;
    stage_start_produced_samples_ = 0;
    latest_produced_samples_ = 0;
    next_evaluation_trained_samples_ =
        config_.evaluation_interval_samples;
    initialized_ = true;
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

int64_t SingleMapTaskController::StageBudget() const {
    switch (stage_) {
        case maze::CURRICULUM_STAGE_8X:
            return config_.stage_8x_sample_budget;
        case maze::CURRICULUM_STAGE_4X:
            return config_.stage_4x_sample_budget;
        case maze::CURRICULUM_STAGE_2X:
            return config_.stage_2x_sample_budget;
        default:
            return 0;
    }
}

int64_t SingleMapTaskController::StageEffectiveBudget() const {
    const int64_t budget = StageBudget();
    if (budget <= 0) return 0;
    return budget - budget % config_.sample_quantum;
}

int64_t SingleMapTaskController::FinalCollectionThreshold() const {
    const int64_t budget = StageEffectiveBudget();
    const int64_t quantum = config_.sample_quantum;
    if (budget <= quantum) return quantum;

    // Collection is observed after a complete producer quantum has been
    // flushed. Reserve one quantum for the current incomplete learner window
    // and one for that bounded observation delay, so the hard stage cap cannot
    // be crossed while preserving whole fragments.
    const int64_t guard = quantum <=
                                  std::numeric_limits<int64_t>::max() / 2
                              ? quantum * 2
                              : std::numeric_limits<int64_t>::max();
    return std::max(quantum, budget > guard ? budget - guard : quantum);
}

double SingleMapTaskController::StageSuccessThreshold() const {
    switch (stage_) {
        case maze::CURRICULUM_STAGE_8X:
            return config_.stage_8x_success_threshold;
        case maze::CURRICULUM_STAGE_4X:
            return config_.stage_4x_success_threshold;
        case maze::CURRICULUM_STAGE_2X:
            return config_.stage_2x_success_threshold;
        default:
            return 1.0;
    }
}

void SingleMapTaskController::StartEvaluation(
    const SingleMapModelIdentity& model,
    int64_t produced_samples,
    bool final_attempt) {
    evaluation_ = EvaluationCampaign{};
    evaluation_.active = true;
    evaluation_.final_attempt = final_attempt;
    evaluation_.model = model;
    evaluation_.produced_samples = produced_samples;
    evaluation_.phase = 0;
    evaluation_.rounds[0].mode = maze::EPISODE_MODE_EVALUATION_ARGMAX;
    evaluation_.rounds[1].mode = maze::EPISODE_MODE_EVALUATION_ARGMAX;
    evaluation_.rounds[2].mode =
        maze::EPISODE_MODE_EVALUATION_STOCHASTIC;
    while (next_evaluation_trained_samples_ <= model.trained_samples) {
        next_evaluation_trained_samples_ +=
            config_.evaluation_interval_samples;
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
    if (!ValidateModel(active_model, error) ||
        produced_samples < latest_produced_samples_ ||
        active_model.trained_samples > produced_samples) {
        if (error.empty()) error = "produced sample counter moved backwards";
        return false;
    }
    latest_produced_samples_ = produced_samples;

    if (stage_ == maze::CURRICULUM_STAGE_COMPLETE ||
        stage_ == maze::CURRICULUM_STAGE_FAILED) {
        plan.continue_task = false;
        plan.curriculum_stage = stage_;
        plan.episode_mode = maze::EPISODE_MODE_UNSPECIFIED;
        plan.model = active_model;
        return true;
    }

    if (!evaluation_.active) {
        const int64_t stage_samples =
            produced_samples - stage_start_produced_samples_;
        if (stage_samples < 0 || stage_samples > StageEffectiveBudget()) {
            error = "single-map stage sample budget was exceeded";
            return false;
        }
        const bool samples_drained =
            active_model.trained_samples == produced_samples;
        const bool final_attempt =
            stage_samples >= FinalCollectionThreshold();
        const bool periodic_evaluation =
            active_model.trained_samples >=
            next_evaluation_trained_samples_;
        if (samples_drained && (periodic_evaluation || final_attempt)) {
            StartEvaluation(active_model, produced_samples, final_attempt);
        }
    }

    plan.continue_task = true;
    plan.curriculum_stage = stage_;
    plan.max_steps = shortest_action_steps_ * StageMultiplier();
    if (evaluation_.active) {
        const auto& round = evaluation_.rounds[evaluation_.phase];
        plan.episode_mode = round.mode;
        plan.model = evaluation_.model;
    } else {
        plan.episode_mode = maze::EPISODE_MODE_TRAINING;
        plan.model = active_model;
    }
    return true;
}

bool SingleMapTaskController::ShouldPauseTrainingCollection(
    const SingleMapModelIdentity& active_model,
    int64_t produced_samples,
    bool& should_pause,
    std::string& error) {
    should_pause = false;
    if (!initialized_) {
        error = "single-map task controller is not initialized";
        return false;
    }
    if (!ValidateModel(active_model, error) ||
        produced_samples < latest_produced_samples_ ||
        active_model.trained_samples > produced_samples) {
        if (error.empty()) {
            error = "single-map training counters are inconsistent";
        }
        return false;
    }
    latest_produced_samples_ = produced_samples;
    if (stage_ == maze::CURRICULUM_STAGE_COMPLETE ||
        stage_ == maze::CURRICULUM_STAGE_FAILED || evaluation_.active) {
        return true;
    }

    const int64_t stage_samples =
        produced_samples - stage_start_produced_samples_;
    if (stage_samples < 0 || stage_samples > StageEffectiveBudget()) {
        error = "single-map stage sample budget was exceeded";
        return false;
    }
    const bool periodic_evaluation =
        active_model.trained_samples >= next_evaluation_trained_samples_;
    const bool final_evaluation =
        stage_samples >= FinalCollectionThreshold();
    const int64_t untrained_samples =
        produced_samples - active_model.trained_samples;
    should_pause =
        (periodic_evaluation || final_evaluation) &&
        (untrained_samples == 0 ||
         untrained_samples >= config_.sample_quantum);
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
    if (evaluation_.active ||
        stage_ == maze::CURRICULUM_STAGE_COMPLETE ||
        stage_ == maze::CURRICULUM_STAGE_FAILED) {
        error = "single-map samples cannot be discarded outside training";
        return false;
    }
    if (latest_produced_samples_ < stage_start_produced_samples_ ||
        sample_count >
            latest_produced_samples_ - stage_start_produced_samples_) {
        error = "single-map discarded samples exceed the current stage";
        return false;
    }

    // Producer freshness is decided asynchronously by the Sample Pool. A
    // rejected fragment was tentatively counted when its transitions were
    // created, so rewind the current-stage ledger before evaluating the next
    // monotonic counter snapshot. Evaluation can only start after this ledger
    // and the Learner trained-sample identity are fully drained.
    latest_produced_samples_ -= sample_count;
    return true;
}

double SingleMapTaskController::SuccessRate(
    const EvaluationRound& round) {
    if (round.agent_episodes <= 0) return 0.0;
    return static_cast<double>(round.successes) /
           static_cast<double>(round.agent_episodes);
}

double SingleMapTaskController::Quantile(
    std::vector<double> values,
    double quantile) {
    if (values.empty()) return std::numeric_limits<double>::infinity();
    std::sort(values.begin(), values.end());
    const double index = quantile * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(index));
    const auto upper = static_cast<std::size_t>(std::ceil(index));
    if (lower == upper) return values[lower];
    const double weight = index - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

void SingleMapTaskController::FinishEvaluation() {
    SingleMapEvaluationRecord record;
    record.curriculum_stage = stage_;
    record.model = evaluation_.model;
    record.final_attempt = evaluation_.final_attempt;
    record.stage_produced_samples =
        evaluation_.produced_samples - stage_start_produced_samples_;
    record.stage_sample_budget = StageBudget();
    record.argmax_round_1_success_rate =
        SuccessRate(evaluation_.rounds[0]);
    record.argmax_round_2_success_rate =
        SuccessRate(evaluation_.rounds[1]);
    record.stochastic_success_rate =
        SuccessRate(evaluation_.rounds[2]);
    std::vector<double> successful_ratios =
        evaluation_.rounds[0].successful_path_ratios;
    successful_ratios.insert(
        successful_ratios.end(),
        evaluation_.rounds[1].successful_path_ratios.begin(),
        evaluation_.rounds[1].successful_path_ratios.end());
    record.successful_path_ratio_median =
        Quantile(successful_ratios, 0.50);
    record.successful_path_ratio_p95 =
        Quantile(successful_ratios, 0.95);

    record.passed =
        record.argmax_round_1_success_rate >= StageSuccessThreshold() &&
        record.argmax_round_2_success_rate >= StageSuccessThreshold();
    if (record.passed && stage_ == maze::CURRICULUM_STAGE_2X) {
        record.passed =
            record.successful_path_ratio_median <=
                config_.final_path_ratio_median_limit &&
            record.successful_path_ratio_p95 <=
                config_.final_path_ratio_p95_limit;
    }
    history_.push_back(record);

    if (record.passed) {
        stage_start_produced_samples_ = evaluation_.produced_samples;
        if (stage_ == maze::CURRICULUM_STAGE_8X) {
            stage_ = maze::CURRICULUM_STAGE_4X;
        } else if (stage_ == maze::CURRICULUM_STAGE_4X) {
            stage_ = maze::CURRICULUM_STAGE_2X;
        } else {
            stage_ = maze::CURRICULUM_STAGE_COMPLETE;
        }
    } else if (evaluation_.final_attempt) {
        stage_ = maze::CURRICULUM_STAGE_FAILED;
    }
    evaluation_ = EvaluationCampaign{};
}

bool SingleMapTaskController::RecordEvaluationEpisode(
    maze::EpisodeMode episode_mode,
    const SingleMapModelIdentity& model,
    const std::vector<SingleMapAgentEvaluation>& agents,
    std::string& error) {
    if (!evaluation_.active || evaluation_.phase < 0 ||
        evaluation_.phase >= 3) {
        error = "no single-map evaluation campaign is active";
        return false;
    }
    auto& round = evaluation_.rounds[evaluation_.phase];
    if (episode_mode != round.mode ||
        !SameModel(model, evaluation_.model)) {
        error = "evaluation mode or pinned model identity changed";
        return false;
    }
    if (agents.size() != static_cast<std::size_t>(config_.agent_num)) {
        error = "evaluation episode must contain every assigned Agent";
        return false;
    }
    for (const auto& agent : agents) {
        if (agent.transition_count <= 0 ||
            agent.transition_count >
                static_cast<int64_t>(shortest_action_steps_) *
                    StageMultiplier()) {
            error = "evaluation transition count is outside the assigned horizon";
            return false;
        }
        ++round.agent_episodes;
        if (agent.success) {
            ++round.successes;
            round.successful_path_ratios.push_back(
                static_cast<double>(agent.transition_count) /
                static_cast<double>(shortest_action_steps_));
        }
    }
    ++round.environment_episodes;
    if (round.environment_episodes >
        config_.evaluation_episodes_per_round) {
        error = "evaluation round received too many environment episodes";
        return false;
    }
    if (round.environment_episodes ==
        config_.evaluation_episodes_per_round) {
        if (evaluation_.phase < 2) {
            ++evaluation_.phase;
        } else {
            FinishEvaluation();
        }
    }
    return true;
}

bool SingleMapTaskController::IsEvaluationActive() const {
    return evaluation_.active;
}

SingleMapTaskSnapshot SingleMapTaskController::GetSnapshot() const {
    SingleMapTaskSnapshot snapshot;
    snapshot.initialized = initialized_;
    snapshot.evaluation_active = evaluation_.active;
    snapshot.curriculum_stage = stage_;
    snapshot.evaluation_episodes_per_round =
        config_.evaluation_episodes_per_round;
    snapshot.stage_produced_samples =
        latest_produced_samples_ - stage_start_produced_samples_;
    snapshot.stage_sample_budget = StageBudget();
    snapshot.next_evaluation_trained_samples =
        next_evaluation_trained_samples_;
    snapshot.complete = stage_ == maze::CURRICULUM_STAGE_COMPLETE;
    snapshot.failed = stage_ == maze::CURRICULUM_STAGE_FAILED;
    if (evaluation_.active) {
        const auto& round = evaluation_.rounds[evaluation_.phase];
        snapshot.evaluation_mode = round.mode;
        snapshot.evaluation_round = evaluation_.phase + 1;
        snapshot.evaluation_episode_in_round = round.environment_episodes;
        snapshot.evaluation_model_version =
            evaluation_.model.model_version;
        snapshot.evaluation_model_trained_samples =
            evaluation_.model.trained_samples;
    }
    if (!history_.empty()) {
        const auto& latest = history_.back();
        snapshot.latest_argmax_round_1_success_rate =
            latest.argmax_round_1_success_rate;
        snapshot.latest_argmax_round_2_success_rate =
            latest.argmax_round_2_success_rate;
        snapshot.latest_stochastic_success_rate =
            latest.stochastic_success_rate;
        snapshot.latest_path_ratio_median =
            latest.successful_path_ratio_median;
        snapshot.latest_path_ratio_p95 =
            latest.successful_path_ratio_p95;
    }
    return snapshot;
}

const std::vector<SingleMapEvaluationRecord>&
SingleMapTaskController::history() const {
    return history_;
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

const char* SingleMapTaskController::ModeName(maze::EpisodeMode mode) {
    switch (mode) {
        case maze::EPISODE_MODE_TRAINING: return "training";
        case maze::EPISODE_MODE_EVALUATION_ARGMAX: return "argmax";
        case maze::EPISODE_MODE_EVALUATION_STOCHASTIC: return "stochastic";
        default: return "unspecified";
    }
}

std::string SingleMapTaskController::ToJson() const {
    const auto snapshot = GetSnapshot();
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\"schema_version\":1"
           << ",\"curriculum_stage\":\"" << StageName(stage_) << "\""
           << ",\"complete\":" << (snapshot.complete ? "true" : "false")
           << ",\"failed\":" << (snapshot.failed ? "true" : "false")
           << ",\"evaluation_active\":"
           << (snapshot.evaluation_active ? "true" : "false")
           << ",\"evaluation_mode\":\""
           << ModeName(snapshot.evaluation_mode) << "\""
           << ",\"stage_produced_samples\":"
           << snapshot.stage_produced_samples
           << ",\"stage_sample_budget\":"
           << snapshot.stage_sample_budget
           << ",\"next_evaluation_trained_samples\":"
           << snapshot.next_evaluation_trained_samples
           << ",\"history\":[";
    for (std::size_t i = 0; i < history_.size(); ++i) {
        const auto& record = history_[i];
        if (i > 0) output << ',';
        output << "{\"stage\":\"" << StageName(record.curriculum_stage)
               << "\",\"model_version\":" << record.model.model_version
               << ",\"model_checksum\":\""
               << record.model.model_checksum
               << "\",\"trained_samples\":"
               << record.model.trained_samples
               << ",\"final_attempt\":"
               << (record.final_attempt ? "true" : "false")
               << ",\"stage_produced_samples\":"
               << record.stage_produced_samples
               << ",\"stage_sample_budget\":"
               << record.stage_sample_budget
               << ",\"argmax_round_1_success_rate\":"
               << record.argmax_round_1_success_rate
               << ",\"argmax_round_2_success_rate\":"
               << record.argmax_round_2_success_rate
               << ",\"stochastic_success_rate\":"
               << record.stochastic_success_rate
               << ",\"successful_path_ratio_median\":";
        if (std::isfinite(record.successful_path_ratio_median)) {
            output << record.successful_path_ratio_median;
        } else {
            output << "null";
        }
        output << ",\"successful_path_ratio_p95\":";
        if (std::isfinite(record.successful_path_ratio_p95)) {
            output << record.successful_path_ratio_p95;
        } else {
            output << "null";
        }
        output << ",\"passed\":"
               << (record.passed ? "true" : "false") << '}';
    }
    output << "]}";
    return output.str();
}
