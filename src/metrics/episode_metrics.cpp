#include "metrics/episode_metrics.h"

#include "task/single_map_task_controller.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace {

bool SnakeCase(const std::string& value) {
    if (value.empty() || value.front() == '_' || value.back() == '_') {
        return false;
    }
    bool previous_underscore = false;
    for (const char character : value) {
        const bool underscore = character == '_';
        if (!underscore && !(character >= 'a' && character <= 'z') &&
            !(character >= '0' && character <= '9')) {
            return false;
        }
        if (underscore && previous_underscore) return false;
        previous_underscore = underscore;
    }
    return true;
}

void FillMetricSchema(common::SchemaIdentity* schema) {
    schema->set_schema_id("maze.metrics.v1");
    schema->set_schema_version(1);
    schema->mutable_canonical_digest()->set_algorithm(
        common::DIGEST_ALGORITHM_SHA256);
    schema->mutable_canonical_digest()->set_hex(
        "2ab9434f8c80b4651b0f51f65f3e94e29b0dd0a55803ed4c7d888f44f4604ce4");
}

void AddDescriptor(training::MetricSnapshot* snapshot,
                   const std::string& field_id,
                   const std::string& label,
                   const std::string& group,
                   const std::string& dimension,
                   const std::string& unit,
                   const std::string& statistic,
                   training::MetricValueKind value_kind,
                   training::MetricAggregationKind aggregation) {
    auto* descriptor = snapshot->add_descriptors();
    descriptor->set_field_id(field_id);
    descriptor->set_label(label);
    descriptor->set_group(group);
    descriptor->set_dimension(dimension);
    descriptor->set_unit(unit);
    descriptor->set_scope("server_pod");
    descriptor->set_statistic(statistic);
    descriptor->set_value_kind(value_kind);
    descriptor->set_owner_component("maze-task-adapter");
    descriptor->set_aggregation_kind(aggregation);
    descriptor->set_window_kind(training::METRIC_WINDOW_KIND_ROLLING);
    FillMetricSchema(descriptor->mutable_schema_identity());
}

void AddMean(training::MetricSnapshot* snapshot,
             const std::string& field_id,
             double sum,
             uint64_t count,
             int64_t timestamp) {
    if (count == 0) return;
    auto* value = snapshot->add_values();
    value->set_field_id(field_id);
    value->set_sum(sum);
    value->set_count(count);
    value->set_value(sum / static_cast<double>(count));
    value->set_window_end_unix_ms(timestamp);
}

void AddGauge(training::MetricSnapshot* snapshot,
              const std::string& field_id,
              double measurement,
              int64_t timestamp) {
    auto* value = snapshot->add_values();
    value->set_field_id(field_id);
    value->set_value(measurement);
    value->set_window_end_unix_ms(timestamp);
}

void AddTaskDescriptor(
    training::MetricSnapshot* snapshot,
    const std::string& field_id,
    const std::string& label,
    const std::string& dimension,
    const std::string& unit,
    const std::string& statistic,
    training::MetricValueKind value_kind,
    training::MetricAggregationKind aggregation,
    training::MetricWindowKind window) {
    auto* descriptor = snapshot->add_descriptors();
    descriptor->set_field_id(field_id);
    descriptor->set_label(label);
    descriptor->set_group(
        field_id.rfind("server.evaluation.", 0) == 0
            ? "episode_success"
            : "training_depth");
    descriptor->set_dimension(dimension);
    descriptor->set_unit(unit);
    descriptor->set_scope("server_pod");
    descriptor->set_statistic(statistic);
    descriptor->set_value_kind(value_kind);
    descriptor->set_owner_component("maze-task-adapter");
    descriptor->set_aggregation_kind(aggregation);
    descriptor->set_window_kind(window);
    FillMetricSchema(descriptor->mutable_schema_identity());
}

double CurriculumMultiplier(maze::CurriculumStage stage) {
    switch (stage) {
        case maze::CURRICULUM_STAGE_8X:
            return 8.0;
        case maze::CURRICULUM_STAGE_4X:
            return 4.0;
        case maze::CURRICULUM_STAGE_2X:
            return 2.0;
        default:
            return 0.0;
    }
}

}  // namespace

EpisodeMetricsWindow::EpisodeMetricsWindow(std::size_t capacity)
    : capacity_(std::max<std::size_t>(1, capacity)) {}

void EpisodeMetricsWindow::Push(Entry entry) {
    entries_.push_back(std::move(entry));
    while (entries_.size() > capacity_) entries_.pop_front();
}

void EpisodeMetricsWindow::AddCompleted(
    std::vector<AgentEpisodeResult> agents) {
    std::lock_guard<std::mutex> lock(mutex_);
    Push(Entry{false, std::move(agents)});
}

void EpisodeMetricsWindow::AddExcluded(
    std::size_t agent_count,
    maze::MazeTerminationReason reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<AgentEpisodeResult> agents(agent_count);
    for (auto& agent : agents) agent.termination_reason = reason;
    Push(Entry{true, std::move(agents)});
}

void EpisodeMetricsWindow::Fill(
    training::MetricSnapshot* snapshot,
    const common::ServiceInstanceIdentity& source,
    uint64_t sequence,
    int64_t timestamp_unix_ms) const {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot->Clear();
    *snapshot->mutable_source() = source;
    snapshot->set_sequence(sequence);
    snapshot->set_timestamp_unix_ms(timestamp_unix_ms);

    AddDescriptor(snapshot, "server.episode.learning_return.mean.v1",
                  "Mean Learning Return", "episode_return", "reward",
                  "reward", "mean", training::METRIC_VALUE_KIND_MEAN,
                  training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
    AddDescriptor(snapshot, "server.episode.learning_return.min.v1",
                  "Min Learning Return", "episode_return", "reward",
                  "reward", "min", training::METRIC_VALUE_KIND_GAUGE,
                  training::METRIC_AGGREGATION_KIND_NOT_MERGEABLE);
    AddDescriptor(snapshot, "server.episode.learning_return.max.v1",
                  "Max Learning Return", "episode_return", "reward",
                  "reward", "max", training::METRIC_VALUE_KIND_GAUGE,
                  training::METRIC_AGGREGATION_KIND_NOT_MERGEABLE);
    AddDescriptor(snapshot, "server.episode.success.agent_rate.v1",
                  "Success Rate", "episode_success", "ratio", "ratio",
                  "rate", training::METRIC_VALUE_KIND_MEAN,
                  training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
    AddDescriptor(snapshot, "server.episode.step.mean.v1", "Episode Step",
                  "episode_success", "count", "step", "mean",
                  training::METRIC_VALUE_KIND_MEAN,
                  training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
    AddDescriptor(snapshot, "server.episode.path_ratio.mean.v1", "Path Ratio",
                  "episode_success", "ratio", "ratio", "mean",
                  training::METRIC_VALUE_KIND_MEAN,
                  training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
    AddDescriptor(snapshot, "server.episode.unique_cells.mean.v1",
                  "Unique Cells", "episode_success", "count", "cell",
                  "mean", training::METRIC_VALUE_KIND_MEAN,
                  training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
    AddDescriptor(snapshot, "server.episode.blocked_move_rate.v1",
                  "Blocked Move Rate", "episode_success", "ratio", "ratio",
                  "rate", training::METRIC_VALUE_KIND_MEAN,
                  training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);

    uint64_t completed_agents = 0;
    uint64_t successes = 0;
    uint64_t reward_transitions = 0;
    uint64_t successful_paths = 0;
    double return_sum = 0.0;
    double return_min = std::numeric_limits<double>::infinity();
    double return_max = -std::numeric_limits<double>::infinity();
    double episode_step_sum = 0.0;
    double path_ratio_sum = 0.0;
    double unique_cells_sum = 0.0;
    double blocked_moves_sum = 0.0;
    std::unordered_map<std::string, double> reward_component_sums;

    for (const auto& entry : entries_) {
        if (entry.excluded) continue;
        for (const auto& agent : entry.agents) {
            ++completed_agents;
            return_sum += agent.episode_return;
            return_min = std::min(return_min, agent.episode_return);
            return_max = std::max(return_max, agent.episode_return);
            episode_step_sum += agent.transition_count;
            unique_cells_sum += agent.unique_cell_count;
            blocked_moves_sum += agent.blocked_move_count;
            reward_transitions += static_cast<uint64_t>(
                std::max<int64_t>(0, agent.transition_count));
            if (agent.success) {
                ++successes;
                if (agent.shortest_action_steps > 0) {
                    path_ratio_sum +=
                        static_cast<double>(agent.transition_count) /
                        static_cast<double>(agent.shortest_action_steps);
                    ++successful_paths;
                }
            }
            for (const auto& item : agent.reward_component_sums) {
                if (SnakeCase(item.first)) {
                    reward_component_sums[item.first] += item.second;
                }
            }
        }
    }

    AddMean(snapshot, "server.episode.learning_return.mean.v1",
            return_sum, completed_agents, timestamp_unix_ms);
    if (completed_agents > 0) {
        AddGauge(snapshot, "server.episode.learning_return.min.v1",
                 return_min, timestamp_unix_ms);
        AddGauge(snapshot, "server.episode.learning_return.max.v1",
                 return_max, timestamp_unix_ms);
    }
    AddMean(snapshot, "server.episode.success.agent_rate.v1",
            static_cast<double>(successes), completed_agents,
            timestamp_unix_ms);
    AddMean(snapshot, "server.episode.step.mean.v1", episode_step_sum,
            completed_agents, timestamp_unix_ms);
    AddMean(snapshot, "server.episode.path_ratio.mean.v1", path_ratio_sum,
            successful_paths, timestamp_unix_ms);
    AddMean(snapshot, "server.episode.unique_cells.mean.v1", unique_cells_sum,
            completed_agents, timestamp_unix_ms);
    AddMean(snapshot, "server.episode.blocked_move_rate.v1",
            blocked_moves_sum, reward_transitions, timestamp_unix_ms);

    for (const auto& item : reward_component_sums) {
        const std::string field_id = "server.reward.component." + item.first +
                                     ".transition_mean.v1";
        AddDescriptor(snapshot, field_id, item.first, "reward_components",
                      "reward", "reward", "transition_mean",
                      training::METRIC_VALUE_KIND_MEAN,
                      training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
        AddMean(snapshot, field_id, item.second, reward_transitions,
                timestamp_unix_ms);
    }
}

void AppendSingleMapTaskMetrics(
    training::MetricSnapshot* snapshot,
    const SingleMapTaskSnapshot& task,
    bool has_completed_evaluation,
    int64_t timestamp_unix_ms) {
    AddTaskDescriptor(
        snapshot, "server.task.curriculum.multiplier.v1", "Curriculum",
        "curriculum_multiplier", "x", "latest",
        training::METRIC_VALUE_KIND_GAUGE,
        training::METRIC_AGGREGATION_KIND_NOT_MERGEABLE,
        training::METRIC_WINDOW_KIND_INSTANT);
    AddTaskDescriptor(
        snapshot, "server.task.stage_produced_samples.v1", "Stage Samples",
        "sample_count", "samples", "total",
        training::METRIC_VALUE_KIND_COUNTER,
        training::METRIC_AGGREGATION_KIND_SUM,
        training::METRIC_WINDOW_KIND_CUMULATIVE);
    AddTaskDescriptor(
        snapshot, "server.task.stage_sample_budget.v1",
        "Stage Sample Budget", "sample_count", "samples", "latest",
        training::METRIC_VALUE_KIND_GAUGE,
        training::METRIC_AGGREGATION_KIND_NOT_MERGEABLE,
        training::METRIC_WINDOW_KIND_INSTANT);
    AddTaskDescriptor(
        snapshot, "server.task.next_evaluation_trained_samples.v1",
        "Next Evaluation", "sample_count", "samples", "latest",
        training::METRIC_VALUE_KIND_GAUGE,
        training::METRIC_AGGREGATION_KIND_NOT_MERGEABLE,
        training::METRIC_WINDOW_KIND_INSTANT);
    AddTaskDescriptor(
        snapshot, "server.evaluation.episode_in_round.v1",
        "Evaluation Episode", "episode_count", "episode", "latest",
        training::METRIC_VALUE_KIND_GAUGE,
        training::METRIC_AGGREGATION_KIND_NOT_MERGEABLE,
        training::METRIC_WINDOW_KIND_INSTANT);

    const char* success_ids[] = {
        "server.evaluation.argmax_round_1_success_rate.v1",
        "server.evaluation.argmax_round_2_success_rate.v1",
        "server.evaluation.stochastic_success_rate.v1",
    };
    const char* success_labels[] = {
        "Argmax Round 1 Success",
        "Argmax Round 2 Success",
        "Stochastic Success",
    };
    for (int index = 0; index < 3; ++index) {
        AddTaskDescriptor(
            snapshot, success_ids[index], success_labels[index], "ratio",
            "ratio", "mean", training::METRIC_VALUE_KIND_GAUGE,
            training::METRIC_AGGREGATION_KIND_NOT_MERGEABLE,
            training::METRIC_WINDOW_KIND_INSTANT);
    }
    AddTaskDescriptor(
        snapshot, "server.evaluation.path_ratio_median.v1",
        "Argmax Path Ratio Median", "ratio", "ratio", "median",
        training::METRIC_VALUE_KIND_GAUGE,
        training::METRIC_AGGREGATION_KIND_NOT_MERGEABLE,
        training::METRIC_WINDOW_KIND_INSTANT);
    AddTaskDescriptor(
        snapshot, "server.evaluation.path_ratio_p95.v1",
        "Argmax Path Ratio p95", "ratio", "ratio", "p95",
        training::METRIC_VALUE_KIND_GAUGE,
        training::METRIC_AGGREGATION_KIND_NOT_MERGEABLE,
        training::METRIC_WINDOW_KIND_INSTANT);

    if (!task.initialized) return;
    AddGauge(snapshot, "server.task.curriculum.multiplier.v1",
             CurriculumMultiplier(task.curriculum_stage), timestamp_unix_ms);
    AddGauge(snapshot, "server.task.stage_produced_samples.v1",
             static_cast<double>(task.stage_produced_samples),
             timestamp_unix_ms);
    AddGauge(snapshot, "server.task.stage_sample_budget.v1",
             static_cast<double>(task.stage_sample_budget),
             timestamp_unix_ms);
    AddGauge(snapshot, "server.task.next_evaluation_trained_samples.v1",
             static_cast<double>(task.next_evaluation_trained_samples),
             timestamp_unix_ms);
    if (task.evaluation_active) {
        AddGauge(snapshot, "server.evaluation.episode_in_round.v1",
                 static_cast<double>(task.evaluation_episode_in_round),
                 timestamp_unix_ms);
    }
    if (!has_completed_evaluation) return;
    AddGauge(snapshot,
             "server.evaluation.argmax_round_1_success_rate.v1",
             task.latest_argmax_round_1_success_rate, timestamp_unix_ms);
    AddGauge(snapshot,
             "server.evaluation.argmax_round_2_success_rate.v1",
             task.latest_argmax_round_2_success_rate, timestamp_unix_ms);
    AddGauge(snapshot,
             "server.evaluation.stochastic_success_rate.v1",
             task.latest_stochastic_success_rate, timestamp_unix_ms);
    if (std::isfinite(task.latest_path_ratio_median)) {
        AddGauge(snapshot, "server.evaluation.path_ratio_median.v1",
                 task.latest_path_ratio_median, timestamp_unix_ms);
    }
    if (std::isfinite(task.latest_path_ratio_p95)) {
        AddGauge(snapshot, "server.evaluation.path_ratio_p95.v1",
                 task.latest_path_ratio_p95, timestamp_unix_ms);
    }
}
