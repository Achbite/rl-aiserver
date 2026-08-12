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
                   training::MetricAggregationKind aggregation,
                   training::MetricWindowKind window =
                       training::METRIC_WINDOW_KIND_ROLLING) {
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
    descriptor->set_window_kind(window);
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

}  // namespace

EpisodeMetricsWindow::EpisodeMetricsWindow(std::size_t capacity)
    : capacity_(std::max<std::size_t>(1, capacity)) {}

void EpisodeMetricsWindow::Push(Entry entry) {
    entries_.push_back(std::move(entry));
    while (entries_.size() > capacity_) entries_.pop_front();
}

void EpisodeMetricsWindow::PushTraining(Entry entry) {
    training_entries_.push_back(std::move(entry));
    while (training_entries_.size() > capacity_) {
        training_entries_.pop_front();
    }
}

void EpisodeMetricsWindow::AddCompleted(
    maze::EpisodeMode episode_mode,
    std::vector<AgentEpisodeResult> agents) {
    std::lock_guard<std::mutex> lock(mutex_);
    Entry entry{false, episode_mode, std::move(agents)};
    if (episode_mode == maze::EPISODE_MODE_TRAINING) {
        PushTraining(entry);
        ++completed_training_episode_count_;
    }
    Push(std::move(entry));
}

void EpisodeMetricsWindow::AddExcluded(
    maze::EpisodeMode episode_mode,
    std::size_t agent_count,
    maze::MazeTerminationReason reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<AgentEpisodeResult> agents(agent_count);
    for (auto& agent : agents) agent.termination_reason = reason;
    Entry entry{true, episode_mode, std::move(agents)};
    Push(std::move(entry));
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

    struct Aggregate {
        uint64_t completed_agents = 0;
        uint64_t successes = 0;
        uint64_t environment_episodes = 0;
        uint64_t any_success_episodes = 0;
        uint64_t all_success_episodes = 0;
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
    };
    const auto add_agents = [](Aggregate& aggregate,
                               const std::vector<AgentEpisodeResult>& agents) {
        for (const auto& agent : agents) {
            ++aggregate.completed_agents;
            aggregate.return_sum += agent.episode_return;
            aggregate.return_min =
                std::min(aggregate.return_min, agent.episode_return);
            aggregate.return_max =
                std::max(aggregate.return_max, agent.episode_return);
            aggregate.episode_step_sum += agent.transition_count;
            aggregate.unique_cells_sum += agent.unique_cell_count;
            aggregate.blocked_moves_sum += agent.blocked_move_count;
            aggregate.reward_transitions += static_cast<uint64_t>(
                std::max<int64_t>(0, agent.transition_count));
            if (agent.success) {
                ++aggregate.successes;
                if (agent.shortest_action_steps > 0) {
                    aggregate.path_ratio_sum +=
                        static_cast<double>(agent.transition_count) /
                        static_cast<double>(agent.shortest_action_steps);
                    ++aggregate.successful_paths;
                }
            }
            for (const auto& item : agent.reward_component_sums) {
                if (SnakeCase(item.first)) {
                    aggregate.reward_component_sums[item.first] += item.second;
                }
            }
        }
    };
    const auto add_entry = [&](Aggregate& aggregate,
                               const std::vector<AgentEpisodeResult>& agents) {
        add_agents(aggregate, agents);
        ++aggregate.environment_episodes;
        bool any_success = false;
        bool all_success = !agents.empty();
        for (const auto& agent : agents) {
            any_success = any_success || agent.success;
            all_success = all_success && agent.success;
        }
        if (any_success) ++aggregate.any_success_episodes;
        if (all_success) ++aggregate.all_success_episodes;
    };

    Aggregate aggregate;
    Aggregate training_aggregate;
    Aggregate latest_training_episode;
    bool has_latest_training_episode = false;
    for (const auto& entry : entries_) {
        if (entry.excluded) continue;
        add_entry(aggregate, entry.agents);
    }
    for (const auto& entry : training_entries_) {
        if (!entry.excluded) {
            add_entry(training_aggregate, entry.agents);
        }
    }
    for (auto entry = training_entries_.rbegin();
         entry != training_entries_.rend(); ++entry) {
        if (!entry->excluded) {
            add_entry(latest_training_episode, entry->agents);
            has_latest_training_episode = true;
            break;
        }
    }

    AddMean(snapshot, "server.episode.learning_return.mean.v1",
            aggregate.return_sum, aggregate.completed_agents,
            timestamp_unix_ms);
    if (aggregate.completed_agents > 0) {
        AddGauge(snapshot, "server.episode.learning_return.min.v1",
                 aggregate.return_min, timestamp_unix_ms);
        AddGauge(snapshot, "server.episode.learning_return.max.v1",
                 aggregate.return_max, timestamp_unix_ms);
    }
    AddMean(snapshot, "server.episode.success.agent_rate.v1",
            static_cast<double>(aggregate.successes),
            aggregate.completed_agents,
            timestamp_unix_ms);
    AddMean(snapshot, "server.episode.step.mean.v1",
            aggregate.episode_step_sum, aggregate.completed_agents,
            timestamp_unix_ms);
    AddMean(snapshot, "server.episode.path_ratio.mean.v1",
            aggregate.path_ratio_sum, aggregate.successful_paths,
            timestamp_unix_ms);
    AddMean(snapshot, "server.episode.unique_cells.mean.v1",
            aggregate.unique_cells_sum, aggregate.completed_agents,
            timestamp_unix_ms);
    AddMean(snapshot, "server.episode.blocked_move_rate.v1",
            aggregate.blocked_moves_sum, aggregate.reward_transitions,
            timestamp_unix_ms);

    for (const auto& item : aggregate.reward_component_sums) {
        const std::string field_id = "server.reward.component." + item.first +
                                     ".transition_mean.v1";
        AddDescriptor(snapshot, field_id, item.first, "reward_components",
                      "reward", "reward", "transition_mean",
                      training::METRIC_VALUE_KIND_MEAN,
                      training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
        AddMean(snapshot, field_id, item.second,
                aggregate.reward_transitions,
                timestamp_unix_ms);
    }

    AddDescriptor(snapshot, "server.training.episode.completed.total.v1",
                  "Completed Training Episodes", "training_depth",
                  "episode_count", "episode", "total",
                  training::METRIC_VALUE_KIND_COUNTER,
                  training::METRIC_AGGREGATION_KIND_SUM,
                  training::METRIC_WINDOW_KIND_CUMULATIVE);
    AddGauge(snapshot, "server.training.episode.completed.total.v1",
             static_cast<double>(completed_training_episode_count_),
             timestamp_unix_ms);
    AddDescriptor(snapshot,
                  "server.training.episode.learning_return.mean.v1",
                  "Mean Training Agent Return", "episode_return",
                  "episode_return", "reward", "mean",
                  training::METRIC_VALUE_KIND_MEAN,
                  training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
    AddMean(snapshot, "server.training.episode.learning_return.mean.v1",
            training_aggregate.return_sum,
            training_aggregate.completed_agents, timestamp_unix_ms);
    AddDescriptor(snapshot,
                  "server.training.episode.success.agent_rate.v1",
                  "Training Agent Success", "episode_success", "ratio",
                  "ratio", "rate", training::METRIC_VALUE_KIND_MEAN,
                  training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
    AddMean(snapshot, "server.training.episode.success.agent_rate.v1",
            static_cast<double>(training_aggregate.successes),
            training_aggregate.completed_agents, timestamp_unix_ms);
    AddDescriptor(snapshot,
                  "server.training.episode.success.any_rate.v1",
                  "Training Any Success", "episode_success", "ratio",
                  "ratio", "rate", training::METRIC_VALUE_KIND_MEAN,
                  training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
    AddMean(snapshot, "server.training.episode.success.any_rate.v1",
            static_cast<double>(training_aggregate.any_success_episodes),
            training_aggregate.environment_episodes, timestamp_unix_ms);
    AddDescriptor(snapshot,
                  "server.training.episode.success.all_rate.v1",
                  "Training All Success", "episode_success", "ratio",
                  "ratio", "rate", training::METRIC_VALUE_KIND_MEAN,
                  training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
    AddMean(snapshot, "server.training.episode.success.all_rate.v1",
            static_cast<double>(training_aggregate.all_success_episodes),
            training_aggregate.environment_episodes, timestamp_unix_ms);
    AddDescriptor(snapshot, "server.training.episode.step.mean.v1",
                  "Training Episode Step", "episode_success",
                  "environment_step", "step", "mean",
                  training::METRIC_VALUE_KIND_MEAN,
                  training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
    AddMean(snapshot, "server.training.episode.step.mean.v1",
            training_aggregate.episode_step_sum,
            training_aggregate.completed_agents, timestamp_unix_ms);
    AddDescriptor(snapshot, "server.training.episode.path_ratio.mean.v1",
                  "Training Path Ratio", "episode_success", "ratio", "1",
                  "mean", training::METRIC_VALUE_KIND_MEAN,
                  training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
    AddMean(snapshot, "server.training.episode.path_ratio.mean.v1",
            training_aggregate.path_ratio_sum,
            training_aggregate.successful_paths, timestamp_unix_ms);
    AddDescriptor(snapshot, "server.training.episode.unique_cells.mean.v1",
                  "Training Unique Cells", "episode_success", "count",
                  "cell", "mean", training::METRIC_VALUE_KIND_MEAN,
                  training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
    AddMean(snapshot, "server.training.episode.unique_cells.mean.v1",
            training_aggregate.unique_cells_sum,
            training_aggregate.completed_agents, timestamp_unix_ms);
    AddDescriptor(snapshot,
                  "server.training.episode.blocked_move_rate.v1",
                  "Training Blocked Move Rate", "episode_success", "ratio",
                  "ratio", "rate", training::METRIC_VALUE_KIND_MEAN,
                  training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
    AddMean(snapshot, "server.training.episode.blocked_move_rate.v1",
            training_aggregate.blocked_moves_sum,
            training_aggregate.reward_transitions, timestamp_unix_ms);
    AddDescriptor(snapshot,
                  "server.training.episode.learning_return.latest_mean.v1",
                  "Latest Training Agent Return", "episode_return",
                  "episode_return", "reward", "latest_mean",
                  training::METRIC_VALUE_KIND_MEAN,
                  training::METRIC_AGGREGATION_KIND_LATEST,
                  training::METRIC_WINDOW_KIND_INSTANT);
    if (has_latest_training_episode) {
        AddMean(snapshot,
                "server.training.episode.learning_return.latest_mean.v1",
                latest_training_episode.return_sum,
                latest_training_episode.completed_agents,
                timestamp_unix_ms);
    }

    std::unordered_map<std::string, bool> training_component_names;
    for (const auto& item : training_aggregate.reward_component_sums) {
        training_component_names[item.first] = true;
    }
    for (const auto& item : latest_training_episode.reward_component_sums) {
        training_component_names[item.first] = true;
    }
    for (const auto& component : training_component_names) {
        const auto& name = component.first;
        const auto aggregate_value =
            training_aggregate.reward_component_sums.find(name);
        const double aggregate_sum =
            aggregate_value == training_aggregate.reward_component_sums.end()
                ? 0.0
                : aggregate_value->second;
        const auto latest_value =
            latest_training_episode.reward_component_sums.find(name);
        const double latest_sum =
            latest_value == latest_training_episode.reward_component_sums.end()
                ? 0.0
                : latest_value->second;

        const std::string prefix =
            "server.training.reward.component." + name;
        AddDescriptor(snapshot, prefix + ".episode_mean.v1", name,
                      "reward_components", "episode_reward",
                      "reward/agent_episode", "mean",
                      training::METRIC_VALUE_KIND_MEAN,
                      training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
        AddMean(snapshot, prefix + ".episode_mean.v1", aggregate_sum,
                training_aggregate.completed_agents, timestamp_unix_ms);
        AddDescriptor(snapshot, prefix + ".transition_mean.v1", name,
                      "reward_components", "transition_reward",
                      "reward/transition", "mean",
                      training::METRIC_VALUE_KIND_MEAN,
                      training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
        AddMean(snapshot, prefix + ".transition_mean.v1", aggregate_sum,
                training_aggregate.reward_transitions, timestamp_unix_ms);
        AddDescriptor(snapshot, prefix + ".latest_episode_mean.v1", name,
                      "reward_components", "episode_reward",
                      "reward/agent_episode", "latest_mean",
                      training::METRIC_VALUE_KIND_MEAN,
                      training::METRIC_AGGREGATION_KIND_LATEST,
                      training::METRIC_WINDOW_KIND_INSTANT);
        if (has_latest_training_episode) {
            AddMean(snapshot, prefix + ".latest_episode_mean.v1", latest_sum,
                    latest_training_episode.completed_agents,
                    timestamp_unix_ms);
        }
    }
}
