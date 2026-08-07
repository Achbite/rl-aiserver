#include "metrics/episode_metrics.h"
#include "task/single_map_task_controller.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << std::endl;
        std::exit(1);
    }
}

bool Near(double left, double right) {
    return std::abs(left - right) < 1e-9;
}

}  // namespace

int main() {
    EpisodeMetricsWindow window(2);
    AgentEpisodeResult success;
    success.episode_return = 12.0;
    success.success = true;
    success.termination_reason =
        maze::MAZE_TERMINATION_REASON_GOAL_REACHED;
    success.transition_count = 2;
    success.shortest_action_steps = 2;
    success.unique_cell_count = 3;
    success.reward_component_sums["goal_reward"] = 10.0;

    AgentEpisodeResult timeout;
    timeout.episode_return = -2.0;
    timeout.termination_reason =
        maze::MAZE_TERMINATION_REASON_TIME_LIMIT;
    timeout.transition_count = 2;
    timeout.shortest_action_steps = 2;
    timeout.unique_cell_count = 2;
    timeout.blocked_move_count = 1;
    timeout.reward_component_sums["goal_reward"] = 0.0;

    window.AddCompleted({success, timeout});
    window.AddExcluded(
        2, maze::MAZE_TERMINATION_REASON_CHAIN_FAILURE);

    common::ServiceInstanceIdentity source;
    source.set_component("rl-aiserver");
    source.set_instance_id("aiserver-test");
    source.set_lifecycle_epoch(1);
    training::MetricSnapshot snapshot;
    window.Fill(&snapshot, source, 7, 1234);

    std::unordered_map<std::string, double> values;
    std::unordered_map<std::string, training::MetricDescriptor> descriptors;
    for (const auto& descriptor : snapshot.descriptors()) {
        Require(descriptors.emplace(descriptor.field_id(), descriptor).second,
                "metric descriptor IDs must be unique");
    }
    for (const auto& value : snapshot.values()) {
        Require(descriptors.find(value.field_id()) != descriptors.end(),
                "every value must have a descriptor");
        values[value.field_id()] = value.value();
    }
    Require(snapshot.source().SerializeAsString() == source.SerializeAsString(),
            "metric source identity");
    Require(snapshot.sequence() == 7 &&
                snapshot.timestamp_unix_ms() == 1234,
            "metric sequence and timestamp");
    Require(Near(values.at("server.episode.learning_return.mean.v1"), 5.0),
            "mean learning return");
    Require(Near(values.at("server.episode.learning_return.min.v1"), -2.0),
            "minimum learning return");
    Require(Near(values.at("server.episode.learning_return.max.v1"), 12.0),
            "maximum learning return");
    Require(Near(values.at("server.episode.success.agent_rate.v1"), 0.5),
            "Agent success rate");
    Require(Near(values.at("server.episode.step.mean.v1"), 2.0),
            "mean Episode Step");
    Require(Near(values.at("server.episode.path_ratio.mean.v1"), 1.0),
            "successful path ratio");
    Require(Near(values.at("server.episode.unique_cells.mean.v1"), 2.5),
            "mean unique cells");
    Require(Near(values.at("server.episode.blocked_move_rate.v1"), 0.25),
            "blocked move rate");
    Require(Near(values.at(
                     "server.reward.component.goal_reward.transition_mean.v1"),
                 2.5),
            "Reward component transition mean");
    Require(descriptors.at("server.episode.success.agent_rate.v1").
                aggregation_kind() ==
                training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN,
            "rates aggregate by weighted mean");

    SingleMapTaskSnapshot task;
    task.initialized = true;
    task.evaluation_active = true;
    task.curriculum_stage = maze::CURRICULUM_STAGE_8X;
    task.evaluation_episode_in_round = 7;
    task.stage_produced_samples = 100352;
    task.stage_sample_budget = 1000000;
    task.next_evaluation_trained_samples = 200000;
    AppendSingleMapTaskMetrics(&snapshot, task, false, 1235);

    descriptors.clear();
    values.clear();
    for (const auto& descriptor : snapshot.descriptors()) {
        Require(descriptors.emplace(descriptor.field_id(), descriptor).second,
                "combined metric descriptor IDs must be unique");
    }
    for (const auto& value : snapshot.values()) {
        Require(descriptors.find(value.field_id()) != descriptors.end(),
                "combined metric values require descriptors");
        values[value.field_id()] = value.value();
    }
    Require(Near(values.at("server.task.curriculum.multiplier.v1"), 8.0),
            "8x curriculum metric");
    Require(Near(values.at("server.task.stage_produced_samples.v1"),
                 100352.0),
            "stage produced samples metric");
    Require(Near(values.at("server.task.stage_sample_budget.v1"),
                 1000000.0),
            "stage sample budget metric");
    Require(Near(values.at(
                     "server.task.next_evaluation_trained_samples.v1"),
                 200000.0),
            "next evaluation metric");
    Require(Near(values.at("server.evaluation.episode_in_round.v1"), 7.0),
            "evaluation episode metric");
    Require(values.find(
                "server.evaluation.argmax_round_1_success_rate.v1") ==
                values.end(),
            "unfinished evaluation must not publish a result");

    task.evaluation_active = false;
    task.latest_argmax_round_1_success_rate = 0.81;
    task.latest_argmax_round_2_success_rate = 0.82;
    task.latest_stochastic_success_rate = 0.75;
    task.latest_path_ratio_median = 1.25;
    task.latest_path_ratio_p95 = 1.75;
    training::MetricSnapshot completed_snapshot;
    AppendSingleMapTaskMetrics(&completed_snapshot, task, true, 1236);
    values.clear();
    for (const auto& value : completed_snapshot.values()) {
        values[value.field_id()] = value.value();
    }
    Require(Near(values.at(
                     "server.evaluation.argmax_round_1_success_rate.v1"),
                 0.81),
            "Argmax round 1 metric");
    Require(Near(values.at(
                     "server.evaluation.argmax_round_2_success_rate.v1"),
                 0.82),
            "Argmax round 2 metric");
    Require(Near(values.at(
                     "server.evaluation.stochastic_success_rate.v1"),
                 0.75),
            "stochastic evaluation metric");
    Require(Near(values.at("server.evaluation.path_ratio_median.v1"),
                 1.25),
            "path-ratio median metric");
    Require(Near(values.at("server.evaluation.path_ratio_p95.v1"), 1.75),
            "path-ratio p95 metric");
    std::cout << "episode_metrics_contract: PASS\n";
    return 0;
}
