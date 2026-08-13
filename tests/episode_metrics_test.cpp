#include "metrics/episode_metrics.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
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

void SetDigest(const std::string& hex, common::ContentDigest* digest) {
    digest->set_algorithm(common::DIGEST_ALGORITHM_SHA256);
    digest->set_hex(hex);
}

common::ContractIdentity Contract() {
    common::ContractIdentity contract;
    contract.set_package_name("rl-contracts");
    contract.set_package_version("0.11.0");
    SetDigest(std::string(64, 'a'), contract.mutable_source_digest());
    SetDigest(std::string(64, 'b'), contract.mutable_artifact_digest());
    contract.set_platform("linux/arm64");
    contract.set_generator_identity(std::string(64, 'c'));
    return contract;
}

common::SchemaIdentity Schema() {
    common::SchemaIdentity schema;
    schema.set_schema_id("maze.metrics.v2");
    schema.set_schema_version(2);
    SetDigest(std::string(64, 'd'), schema.mutable_canonical_digest());
    return schema;
}

training::GetMetricBatchReq Request(
    const common::ContractIdentity& contract,
    const common::ServiceInstanceIdentity& source,
    const common::ServiceInstanceIdentity& consumer) {
    training::GetMetricBatchReq request;
    *request.mutable_contract() = contract;
    *request.mutable_consumer() = consumer;
    *request.mutable_cursor()->mutable_source() = source;
    request.set_max_events(32);
    request.set_max_bytes(1 << 20);
    request.set_wait_timeout_ms(0);
    return request;
}

}  // namespace

int main() {
    EpisodeMetricsWindow window(4);
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

    window.AddCompleted(maze::EPISODE_MODE_TRAINING, {success, timeout});
    window.AddExcluded(
        maze::EPISODE_MODE_TRAINING, 2,
        maze::MAZE_TERMINATION_REASON_CHAIN_FAILURE);

    common::ServiceInstanceIdentity source;
    source.set_component("rl-aiserver");
    source.set_instance_id("aiserver-test");
    source.set_lifecycle_epoch(1);
    training::MetricSnapshot snapshot;
    window.Fill(&snapshot, source, 7, 1234);

    std::unordered_map<std::string, double> values;
    std::unordered_map<std::string, training::MetricValue> metric_values;
    std::unordered_map<std::string, training::MetricDescriptor> descriptors;
    for (const auto& descriptor : snapshot.descriptors()) {
        Require(descriptors.emplace(descriptor.field_id(), descriptor).second,
                "metric descriptor IDs must be unique");
    }
    for (const auto& value : snapshot.values()) {
        Require(descriptors.find(value.field_id()) != descriptors.end(),
                "every value must have a descriptor");
        values[value.field_id()] = value.value();
        metric_values[value.field_id()] = value;
    }
    Require(snapshot.source().SerializeAsString() == source.SerializeAsString(),
            "metric source identity");
    Require(snapshot.sequence() == 7 &&
                snapshot.timestamp_unix_ms() == 1234,
            "metric sequence and timestamp");
    Require(Near(values.at("server.episode.learning_return.mean.v1"),
                 5.0),
            "single-link mean uses completed training Agent Episodes");
    Require(Near(values.at("server.episode.learning_return.min.v1"), -2.0),
            "minimum learning return");
    Require(Near(values.at("server.episode.learning_return.max.v1"), 12.0),
            "maximum learning return");
    Require(Near(values.at("server.episode.success.agent_rate.v1"), 0.5),
            "Agent success rate");
    Require(Near(values.at("server.episode.step.mean.v1"), 2.0),
            "mean Episode Step");
    Require(Near(values.at("server.episode.path_ratio.mean.v1"), 1.0),
            "path ratio uses completed successful paths");
    Require(Near(values.at("server.episode.unique_cells.mean.v1"), 2.5),
            "mean unique cells");
    Require(Near(values.at("server.episode.blocked_move_rate.v1"), 0.25),
            "blocked move rate");
    Require(Near(values.at(
                     "server.reward.component.goal_reward.transition_mean.v1"),
                 2.5),
            "Reward component transition mean");
    Require(Near(values.at(
                     "server.training.episode.learning_return.mean.v1"),
                 5.0),
            "training return excludes evaluation");
    Require(Near(values.at(
                     "server.training.episode.learning_return.latest_mean.v1"),
                 5.0),
            "latest training return");
    Require(Near(values.at(
                     "server.training.episode.success.agent_rate.v1"),
                 0.5),
            "training success excludes evaluation");
    Require(Near(values.at(
                     "server.training.episode.success.any_rate.v1"),
                 1.0),
            "training any-success uses Environment Episodes");
    Require(Near(values.at(
                     "server.training.episode.success.all_rate.v1"),
                 0.0),
            "training all-success uses Environment Episodes");
    Require(Near(values.at(
                     "server.training.reward.component.goal_reward."
                     "episode_mean.v1"),
                 5.0),
            "training component mean uses completed Agent Episodes");
    Require(Near(values.at(
                     "server.training.reward.component.goal_reward."
                     "transition_mean.v1"),
                 2.5),
            "training component density uses transitions");
    const auto& training_return = metric_values.at(
        "server.training.episode.learning_return.mean.v1");
    Require(Near(training_return.sum(), 10.0) && training_return.count() == 2,
            "training return preserves raw sum/count");
    const auto& training_component = metric_values.at(
        "server.training.reward.component.goal_reward.episode_mean.v1");
    Require(Near(training_component.sum(), 10.0) &&
                training_component.count() == 2,
            "component Episode mean preserves raw sum/count");
    Require(Near(values.at("server.training.episode.completed.total.v1"), 1.0),
            "training Environment Episode counter excludes evaluation");
    Require(descriptors.at("server.episode.success.agent_rate.v1").
                aggregation_kind() ==
                training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN,
            "rates aggregate by weighted mean");

    for (const auto& item : values) {
        Require(item.first.rfind("server.evaluation.", 0) != 0 &&
                    item.first.rfind("server.task.", 0) != 0,
                "training metrics must not publish evaluation/task gates");
    }

    common::ServiceInstanceIdentity event_source;
    event_source.set_component("rl-aiserver");
    event_source.set_instance_id("aiserver-event-source");
    event_source.set_lifecycle_epoch(3);
    common::ServiceInstanceIdentity consumer;
    consumer.set_component("rl-learner");
    consumer.set_instance_id("learner-event-relay");
    consumer.set_lifecycle_epoch(9);
    const auto contract = Contract();
    MetricEventJournal journal(
        contract, Schema(), event_source, 2,
        MetricEventJournal::kDefaultByteCapacity,
        std::chrono::milliseconds(0));
    auto request = Request(contract, event_source, consumer);
    training::GetMetricBatchRsp empty;
    journal.Get(request, empty);
    Require(empty.result() == training::METRIC_BATCH_RESULT_WAIT &&
                !empty.has_batch(),
            "no-data returns WAIT and omits a false zero batch");

    training::EpisodeMetricFact episode;
    episode.set_task_id("maze.fixed.single-map.v1");
    episode.set_environment_instance_id("env-0");
    episode.set_episode_id("episode-1");
    auto* agent = episode.add_agents();
    agent->set_agent_id(1);
    agent->set_episode_return(10.0);
    agent->set_transition_count(4);
    agent->set_success(true);
    agent->set_termination_reason("MAZE_TERMINATION_REASON_GOAL_REACHED");
    agent->set_behavior_model_lineage_id("lineage-1");
    journal.AppendEpisode(episode, 1000);

    training::GetMetricBatchRsp delivered;
    journal.Get(request, delivered);
    Require(delivered.ret_code() == 0 &&
                delivered.result() ==
                    training::METRIC_BATCH_RESULT_DELIVERED &&
                delivered.has_batch() &&
                delivered.batch().batch_sequence() == 1 &&
                delivered.batch().events_size() == 1 &&
                delivered.batch().events(0).event_sequence() == 1 &&
                delivered.batch().events(0).episode().episode_id() ==
                    "episode-1",
            "committed Episode fact is delivered with exact source sequence");
    training::GetMetricBatchRsp replay;
    journal.Get(request, replay);
    Require(replay.batch().SerializeAsString() ==
                delivered.batch().SerializeAsString(),
            "same unacknowledged cursor replays exact batch identity");

    training::AckMetricBatchReq ack;
    *ack.mutable_contract() = contract;
    *ack.mutable_consumer() = consumer;
    *ack.mutable_cursor()->mutable_source() = event_source;
    ack.mutable_cursor()->set_acknowledged_batch_sequence(1);
    ack.mutable_cursor()->set_acknowledged_event_sequence(1);
    *ack.mutable_cursor()->mutable_acknowledged_batch_digest() =
        delivered.batch().batch_digest();
    training::AckMetricBatchRsp acknowledged;
    journal.Ack(ack, acknowledged);
    Require(acknowledged.result() ==
                training::METRIC_BATCH_ACK_RESULT_APPLIED &&
                acknowledged.committed_cursor().acknowledged_event_sequence() ==
                    1,
            "exact metric batch ACK commits the event cursor");
    training::AckMetricBatchRsp duplicate_ack;
    journal.Ack(ack, duplicate_ack);
    Require(duplicate_ack.result() ==
                training::METRIC_BATCH_ACK_RESULT_ALREADY_APPLIED,
            "metric batch ACK retry is idempotent");

    request.mutable_cursor()->CopyFrom(ack.cursor());
    journal.Finalize(2000);
    training::GetMetricBatchRsp final_batch;
    journal.Get(request, final_batch);
    Require(final_batch.result() ==
                training::METRIC_BATCH_RESULT_DELIVERED &&
                final_batch.batch().heartbeat() &&
                final_batch.batch().source_final() &&
                final_batch.batch().final_event_sequence() == 1,
            "final source emits one acknowledged final heartbeat");
    training::AckMetricBatchReq final_ack;
    *final_ack.mutable_contract() = contract;
    *final_ack.mutable_consumer() = consumer;
    *final_ack.mutable_cursor()->mutable_source() = event_source;
    final_ack.mutable_cursor()->set_acknowledged_batch_sequence(
        final_batch.batch().batch_sequence());
    final_ack.mutable_cursor()->set_acknowledged_event_sequence(1);
    *final_ack.mutable_cursor()->mutable_acknowledged_batch_digest() =
        final_batch.batch().batch_digest();
    training::AckMetricBatchRsp final_acked;
    journal.Ack(final_ack, final_acked);
    request.mutable_cursor()->CopyFrom(final_ack.cursor());
    training::GetMetricBatchRsp final_state;
    journal.Get(request, final_state);
    Require(final_state.result() == training::METRIC_BATCH_RESULT_FINAL &&
                !final_state.has_batch(),
            "final ACK converges to terminal source state without new batches");

    MetricEventJournal bounded(
        contract, Schema(), event_source, 2,
        MetricEventJournal::kDefaultByteCapacity,
        std::chrono::milliseconds(0));
    bounded.AppendEpisode(episode, 1);
    episode.set_episode_id("episode-2");
    bounded.AppendEpisode(episode, 2);
    episode.set_episode_id("episode-3");
    bounded.AppendEpisode(episode, 3);
    auto gap_request = Request(contract, event_source, consumer);
    training::GetMetricBatchRsp gap;
    bounded.Get(gap_request, gap);
    Require(gap.has_batch() && gap.batch().has_gap() &&
                gap.batch().gap().first_unavailable_event_sequence() == 1 &&
                gap.batch().gap().last_unavailable_event_sequence() == 1 &&
                gap.batch().gap().oldest_available_event_sequence() == 2,
            "bounded journal overflow is explicit as a sequence gap");

    MetricEventJournal cadenced(
        contract, Schema(), event_source, 4,
        MetricEventJournal::kDefaultByteCapacity,
        std::chrono::milliseconds(20));
    episode.set_episode_id("cadenced-episode");
    Require(cadenced.AppendEpisode(episode, 10),
            "cadenced event is accepted");
    auto cadenced_request = Request(contract, event_source, consumer);
    training::GetMetricBatchRsp before_flush;
    cadenced.Get(cadenced_request, before_flush);
    Require(before_flush.result() == training::METRIC_BATCH_RESULT_WAIT &&
                !before_flush.has_batch(),
            "event remains unsealed before the producer flush window");
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    training::GetMetricBatchRsp after_flush;
    cadenced.Get(cadenced_request, after_flush);
    Require(after_flush.result() == training::METRIC_BATCH_RESULT_DELIVERED &&
                after_flush.batch().events_size() == 1,
            "producer seals the event after the flush window");

    MetricEventJournal pinning(
        contract, Schema(), event_source, 4,
        MetricEventJournal::kDefaultByteCapacity,
        std::chrono::milliseconds(0));
    Require(pinning.AppendEpisode(episode, 20),
            "pinning fixture event is accepted");
    auto invalid_cursor = Request(contract, event_source, consumer);
    invalid_cursor.mutable_cursor()->set_acknowledged_event_sequence(1);
    training::GetMetricBatchRsp cursor_rejected;
    pinning.Get(invalid_cursor, cursor_rejected);
    Require(cursor_rejected.result() ==
                training::METRIC_BATCH_RESULT_REJECTED_CURSOR,
            "invalid initial cursor is rejected");
    common::ServiceInstanceIdentity second_consumer = consumer;
    second_consumer.set_instance_id("learner-event-relay-2");
    auto second_request = Request(contract, event_source, second_consumer);
    training::GetMetricBatchRsp second_delivered;
    pinning.Get(second_request, second_delivered);
    Require(second_delivered.result() ==
                training::METRIC_BATCH_RESULT_DELIVERED,
            "a rejected first request cannot pin the consumer identity");

    MetricEventJournal frozen(
        contract, Schema(), event_source, 2,
        MetricEventJournal::kDefaultByteCapacity,
        std::chrono::milliseconds(0));
    episode.set_episode_id("frozen-1");
    frozen.AppendEpisode(episode, 30);
    auto frozen_request = Request(contract, event_source, consumer);
    training::GetMetricBatchRsp frozen_first;
    frozen.Get(frozen_request, frozen_first);
    episode.set_episode_id("frozen-2");
    frozen.AppendEpisode(episode, 31);
    episode.set_episode_id("frozen-3");
    frozen.AppendEpisode(episode, 32);
    frozen.Finalize(40);
    training::GetMetricBatchRsp frozen_replay;
    frozen.Get(frozen_request, frozen_replay);
    Require(frozen_replay.batch().SerializeAsString() ==
                frozen_first.batch().SerializeAsString(),
            "overflow and finalize preserve an already delivered batch");

    MetricEventJournal final_gap(
        contract, Schema(), event_source, 2,
        MetricEventJournal::kDefaultByteCapacity,
        std::chrono::milliseconds(0));
    for (int index = 1; index <= 4; ++index) {
        episode.set_episode_id("final-gap-" + std::to_string(index));
        final_gap.AppendEpisode(episode, 50 + index);
    }
    final_gap.Finalize(60);
    auto final_gap_request = Request(contract, event_source, consumer);
    training::GetMetricBatchRsp first_gap;
    final_gap.Get(final_gap_request, first_gap);
    Require(first_gap.batch().has_gap() &&
                first_gap.batch().gap().last_unavailable_event_sequence() == 2 &&
                !first_gap.batch().source_final(),
            "an intermediate overflow gap cannot claim source final");

    MetricEventJournal byte_bounded(
        contract, Schema(), event_source, 8, 1,
        std::chrono::milliseconds(0));
    episode.set_episode_id("byte-overflow");
    byte_bounded.AppendEpisode(episode, 70);
    auto byte_request = Request(contract, event_source, consumer);
    training::GetMetricBatchRsp byte_gap;
    byte_bounded.Get(byte_request, byte_gap);
    Require(byte_gap.has_batch() && byte_gap.batch().has_gap() &&
                byte_gap.oldest_available_event_sequence() == 2,
            "journal byte capacity produces an explicit gap");

    MetricEventJournal heartbeat_order(
        contract, Schema(), event_source, 4,
        MetricEventJournal::kDefaultByteCapacity,
        std::chrono::milliseconds(0));
    auto heartbeat_request = Request(contract, event_source, consumer);
    heartbeat_request.set_wait_timeout_ms(1);
    training::GetMetricBatchRsp live_heartbeat;
    heartbeat_order.Get(heartbeat_request, live_heartbeat);
    Require(live_heartbeat.has_batch() &&
                live_heartbeat.batch().heartbeat() &&
                !live_heartbeat.batch().source_final(),
            "idle long poll produces an acknowledgeable live heartbeat");
    episode.set_episode_id("after-heartbeat");
    heartbeat_order.AppendEpisode(episode, 1);
    training::AckMetricBatchReq heartbeat_ack;
    *heartbeat_ack.mutable_contract() = contract;
    *heartbeat_ack.mutable_consumer() = consumer;
    *heartbeat_ack.mutable_cursor()->mutable_source() = event_source;
    heartbeat_ack.mutable_cursor()->set_acknowledged_batch_sequence(
        live_heartbeat.batch().batch_sequence());
    heartbeat_ack.mutable_cursor()->set_acknowledged_event_sequence(0);
    *heartbeat_ack.mutable_cursor()->mutable_acknowledged_batch_digest() =
        live_heartbeat.batch().batch_digest();
    training::AckMetricBatchRsp heartbeat_applied;
    heartbeat_order.Ack(heartbeat_ack, heartbeat_applied);
    heartbeat_request.mutable_cursor()->CopyFrom(heartbeat_ack.cursor());
    heartbeat_request.set_wait_timeout_ms(0);
    training::GetMetricBatchRsp after_heartbeat;
    heartbeat_order.Get(heartbeat_request, after_heartbeat);
    Require(after_heartbeat.has_batch() &&
                after_heartbeat.batch().events_size() == 1 &&
                after_heartbeat.batch().events(0).committed_at_unix_ms() >
                    live_heartbeat.batch().event_time_watermark_unix_ms(),
            "an event appended behind a pending heartbeat is newer than its watermark");
    std::cout << "episode_metrics_contract: PASS\n";
    return 0;
}
