#include "metrics/episode_metrics.h"

#include "task/single_map_task_controller.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <openssl/evp.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <thread>
#include <utility>

namespace {

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

std::string DeterministicBytes(
    const google::protobuf::MessageLite& message) {
    std::string output(message.ByteSizeLong(), '\0');
    google::protobuf::io::ArrayOutputStream array(
        output.data(), static_cast<int>(output.size()));
    google::protobuf::io::CodedOutputStream coded(&array);
    coded.SetSerializationDeterministic(true);
    if (!message.SerializeToCodedStream(&coded) || coded.HadError()) {
        return "";
    }
    output.resize(static_cast<std::size_t>(coded.ByteCount()));
    return output;
}

std::string Sha256(const std::string& payload) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context) return "";
    bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
              EVP_DigestUpdate(
                  context, payload.data(), payload.size()) == 1;
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int size = 0;
    if (ok) ok = EVP_DigestFinal_ex(context, digest, &size) == 1;
    EVP_MD_CTX_free(context);
    if (!ok) return "";
    static constexpr char kHex[] = "0123456789abcdef";
    std::string output(size * 2, '0');
    for (unsigned int index = 0; index < size; ++index) {
        output[index * 2] = kHex[digest[index] >> 4];
        output[index * 2 + 1] = kHex[digest[index] & 0x0f];
    }
    return output;
}

bool SameIdentity(const common::ServiceInstanceIdentity& left,
                  const common::ServiceInstanceIdentity& right) {
    return left.component() == right.component() &&
           left.instance_id() == right.instance_id() &&
           left.lifecycle_epoch() == right.lifecycle_epoch();
}

bool SameDigest(const common::ContentDigest& left,
                const common::ContentDigest& right) {
    return left.algorithm() == right.algorithm() &&
           left.hex() == right.hex();
}

}  // namespace

MetricEventJournal::MetricEventJournal(
    common::ContractIdentity contract,
    common::SchemaIdentity schema,
    common::ServiceInstanceIdentity source,
    std::size_t capacity,
    std::size_t byte_capacity,
    std::chrono::milliseconds flush_interval)
    : contract_(std::move(contract)),
      schema_(std::move(schema)),
      source_(std::move(source)),
      capacity_(capacity),
      byte_capacity_(byte_capacity),
      flush_interval_(flush_interval),
      last_batch_created_at_(std::chrono::steady_clock::now()) {
    *committed_cursor_.mutable_source() = source_;
}

MetricEventJournal::AppendResult MetricEventJournal::AppendEpisode(
    training::EpisodeMetricFact fact,
    int64_t observed_at_unix_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    AppendResult result;
    if (source_final_) {
        result.code = AppendResult::Code::SourceFinal;
        return result;
    }
    training::MetricEvent event;
    *event.mutable_contract() = contract_;
    *event.mutable_schema_identity() = schema_;
    *event.mutable_source() = source_;
    if (last_event_observed_at_unix_ms_ &&
        observed_at_unix_ms < *last_event_observed_at_unix_ms_) {
        result.wall_clock_regressed = true;
        result.previous_observed_at_unix_ms =
            *last_event_observed_at_unix_ms_;
    }
    event.set_event_sequence(next_event_sequence_++);
    event.set_observed_at_unix_ms(observed_at_unix_ms);
    *event.mutable_episode() = std::move(fact);
    last_event_observed_at_unix_ms_ = observed_at_unix_ms;
    event_bytes_ += event.ByteSizeLong();
    events_.push_back(std::move(event));
    event_enqueued_at_.push_back(std::chrono::steady_clock::now());
    while (events_.size() > capacity_ || event_bytes_ > byte_capacity_) {
        event_bytes_ -= events_.front().ByteSizeLong();
        events_.pop_front();
        event_enqueued_at_.pop_front();
    }
    changed_.notify_all();
    return result;
}

void MetricEventJournal::Finalize() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (source_final_) return;
    source_final_ = true;
    final_event_sequence_ = next_event_sequence_ - 1;
    changed_.notify_all();
}

bool MetricEventJournal::WaitForFinalAcknowledgement(
    std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    // A source that never had a consumer has no established delivery stream
    // to drain. Once a consumer is pinned, however, shutdown must keep the
    // service reachable until that consumer durably ACKs the final batch.
    if (!consumer_ || final_batch_acknowledged_) return true;
    if (timeout <= std::chrono::milliseconds(0)) return false;
    return changed_.wait_for(lock, timeout, [this] {
        return final_batch_acknowledged_;
    });
}

bool MetricEventJournal::ValidContract(
    const common::ContractIdentity& contract) const {
    return contract.SerializeAsString() == contract_.SerializeAsString();
}

bool MetricEventJournal::ValidConsumer(
    const common::ServiceInstanceIdentity& consumer) const {
    return !consumer.component().empty() &&
           !consumer.instance_id().empty() &&
           consumer.lifecycle_epoch() > 0;
}

bool MetricEventJournal::SameConsumer(
    const common::ServiceInstanceIdentity& consumer) const {
    return consumer_ && SameIdentity(*consumer_, consumer);
}

bool MetricEventJournal::CursorMatchesCommitted(
    const training::MetricBatchCursor& cursor) const {
    if (!SameIdentity(cursor.source(), source_)) return false;
    if (cursor.acknowledged_batch_sequence() !=
            committed_cursor_.acknowledged_batch_sequence() ||
        cursor.acknowledged_event_sequence() !=
            committed_cursor_.acknowledged_event_sequence()) {
        return false;
    }
    if (cursor.acknowledged_batch_sequence() == 0) {
        return cursor.acknowledged_batch_digest().hex().empty();
    }
    return SameDigest(cursor.acknowledged_batch_digest(),
                      committed_cursor_.acknowledged_batch_digest());
}

bool MetricEventJournal::ReadyToSeal(
    std::chrono::steady_clock::time_point now) const {
    if (source_final_) return true;
    const uint64_t next_requested =
        committed_cursor_.acknowledged_event_sequence() + 1;
    const uint64_t oldest = events_.empty()
        ? next_event_sequence_
        : events_.front().event_sequence();
    if (next_requested < oldest) return true;

    std::size_t pending_count = 0;
    std::size_t pending_bytes = 0;
    for (std::size_t index = 0; index < events_.size(); ++index) {
        if (events_[index].event_sequence() < next_requested) continue;
        if (pending_count == 0 &&
            now >= event_enqueued_at_[index] + flush_interval_) {
            return true;
        }
        ++pending_count;
        pending_bytes += events_[index].ByteSizeLong();
        if (pending_count >= 1024 ||
            pending_bytes >= 16U * 1024U * 1024U) {
            return true;
        }
    }
    if (pending_count > 0) return false;
    return now >= last_batch_created_at_ + flush_interval_;
}

std::chrono::steady_clock::time_point
MetricEventJournal::NextSealDeadline() const {
    const uint64_t next_requested =
        committed_cursor_.acknowledged_event_sequence() + 1;
    for (std::size_t index = 0; index < events_.size(); ++index) {
        if (events_[index].event_sequence() >= next_requested) {
            return event_enqueued_at_[index] + flush_interval_;
        }
    }
    return last_batch_created_at_ + flush_interval_;
}

void MetricEventJournal::FillAvailability(
    training::GetMetricBatchRsp& response) const {
    *response.mutable_producer() = source_;
    response.set_oldest_available_event_sequence(
        events_.empty() ? next_event_sequence_ :
                          events_.front().event_sequence());
    response.set_latest_available_event_sequence(next_event_sequence_ - 1);
}

void MetricEventJournal::FillAvailability(
    training::AckMetricBatchRsp& response) const {
    *response.mutable_producer() = source_;
    response.set_oldest_available_event_sequence(
        events_.empty() ? next_event_sequence_ :
                          events_.front().event_sequence());
    response.set_latest_available_event_sequence(next_event_sequence_ - 1);
}

std::string MetricEventJournal::BatchDigest(
    const training::MetricBatch& batch) {
    training::MetricBatch canonical = batch;
    canonical.clear_batch_digest();
    return Sha256(DeterministicBytes(canonical));
}

bool MetricEventJournal::BuildPendingBatch(
    const training::GetMetricBatchReq& request,
    int64_t now_unix_ms,
    std::string& error) {
    training::MetricBatch batch;
    *batch.mutable_contract() = contract_;
    *batch.mutable_schema_identity() = schema_;
    *batch.mutable_source() = source_;
    batch.set_batch_sequence(next_batch_sequence_);
    batch.set_created_at_unix_ms(now_unix_ms);

    const uint64_t next_requested =
        committed_cursor_.acknowledged_event_sequence() + 1;
    const uint64_t oldest =
        events_.empty() ? next_event_sequence_ :
                          events_.front().event_sequence();
    if (next_requested < oldest) {
        batch.set_first_event_sequence(next_requested);
        batch.set_last_event_sequence(oldest - 1);
        auto* gap = batch.mutable_gap();
        gap->set_first_unavailable_event_sequence(next_requested);
        gap->set_last_unavailable_event_sequence(oldest - 1);
        gap->set_oldest_available_event_sequence(oldest);
        gap->set_reason("bounded metric event journal overflow");
    } else {
        const std::size_t max_events = request.max_events();
        const int64_t max_bytes = request.max_bytes();
        bool next_event_exists = false;
        for (const auto& event : events_) {
            if (event.event_sequence() < next_requested) continue;
            next_event_exists = true;
            if (batch.events_size() >= static_cast<int>(max_events)) break;
            *batch.add_events() = event;
            if (batch.ByteSizeLong() > static_cast<std::size_t>(max_bytes)) {
                batch.mutable_events()->RemoveLast();
                break;
            }
        }
        if (batch.events_size() > 0) {
            batch.set_first_event_sequence(
                batch.events(0).event_sequence());
            batch.set_last_event_sequence(
                batch.events(batch.events_size() - 1).event_sequence());
        } else if (next_event_exists) {
            error = "max_bytes is smaller than the next metric event";
            return false;
        } else {
            batch.set_heartbeat(true);
        }
    }
    const bool final_event_batch =
        batch.events_size() > 0 &&
        batch.last_event_sequence() == final_event_sequence_;
    const bool final_gap_batch =
        batch.has_gap() &&
        batch.last_event_sequence() == final_event_sequence_;
    const bool final_heartbeat =
        batch.heartbeat() &&
        committed_cursor_.acknowledged_event_sequence() ==
            final_event_sequence_;
    if (source_final_ &&
        (final_event_batch || final_gap_batch || final_heartbeat)) {
        batch.set_source_final(true);
        batch.set_final_event_sequence(final_event_sequence_);
    }
    batch.mutable_batch_digest()->set_algorithm(
        common::DIGEST_ALGORITHM_SHA256);
    batch.mutable_batch_digest()->set_hex(BatchDigest(batch));
    if (batch.ByteSizeLong() > static_cast<std::size_t>(request.max_bytes())) {
        error = "max_bytes is smaller than the next metric batch";
        return false;
    }
    pending_batch_ = std::move(batch);
    ++next_batch_sequence_;
    last_batch_created_at_ = std::chrono::steady_clock::now();
    error.clear();
    return true;
}

void MetricEventJournal::Get(
    const training::GetMetricBatchReq& request,
    training::GetMetricBatchRsp& response) {
    std::unique_lock<std::mutex> lock(mutex_);
    response.Clear();
    FillAvailability(response);
    if (!ValidContract(request.contract()) ||
        !ValidConsumer(request.consumer())) {
        response.set_ret_code(-1);
        response.set_result(
            training::METRIC_BATCH_RESULT_REJECTED_IDENTITY);
        response.set_message("metric consumer contract or identity is invalid");
        return;
    }
    if (consumer_ && !SameConsumer(request.consumer())) {
        response.set_ret_code(-1);
        response.set_result(
            training::METRIC_BATCH_RESULT_REJECTED_IDENTITY);
        response.set_message("metric journal is pinned to another consumer");
        return;
    }
    if (!CursorMatchesCommitted(request.cursor())) {
        response.set_ret_code(-1);
        response.set_result(
            training::METRIC_BATCH_RESULT_REJECTED_CURSOR);
        response.set_message("metric cursor does not match committed cursor");
        return;
    }
    if (request.max_events() == 0 || request.max_events() > 1024 ||
        request.max_bytes() <= 0 || request.max_bytes() > 16 * 1024 * 1024 ||
        request.wait_timeout_ms() < 0 || request.wait_timeout_ms() > 5000) {
        response.set_ret_code(-1);
        response.set_result(
            training::METRIC_BATCH_RESULT_REJECTED_INVALID);
        response.set_message("metric batch limits are invalid");
        return;
    }
    if (!consumer_) consumer_ = request.consumer();
    if (pending_batch_) {
        response.set_ret_code(0);
        response.set_result(training::METRIC_BATCH_RESULT_DELIVERED);
        *response.mutable_batch() = *pending_batch_;
        FillAvailability(response);
        return;
    }
    if (source_final_ && final_batch_acknowledged_) {
        response.set_ret_code(0);
        response.set_result(training::METRIC_BATCH_RESULT_FINAL);
        response.set_message("metric source final batch is acknowledged");
        return;
    }

    const uint64_t next_requested =
        committed_cursor_.acknowledged_event_sequence() + 1;
    const uint64_t oldest = events_.empty()
        ? next_event_sequence_
        : events_.front().event_sequence();
    const bool has_uncommitted_event =
        !events_.empty() &&
        events_.back().event_sequence() >= next_requested;
    const bool has_gap = next_requested < oldest;
    if (!source_final_ && !has_uncommitted_event && !has_gap &&
        request.wait_timeout_ms() == 0) {
        response.set_ret_code(0);
        response.set_result(training::METRIC_BATCH_RESULT_WAIT);
        response.set_message("no metric event is currently available");
        return;
    }

    const auto request_deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(request.wait_timeout_ms());
    while (!ReadyToSeal(std::chrono::steady_clock::now()) &&
           request.wait_timeout_ms() > 0 &&
           std::chrono::steady_clock::now() < request_deadline) {
        changed_.wait_until(
            lock, std::min(request_deadline, NextSealDeadline()));
    }
    if (!ReadyToSeal(std::chrono::steady_clock::now())) {
        response.set_ret_code(0);
        response.set_result(training::METRIC_BATCH_RESULT_WAIT);
        response.set_message("metric flush window has not closed");
        return;
    }
    const int64_t now_unix_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    std::string build_error;
    if (!BuildPendingBatch(request, now_unix_ms, build_error)) {
        response.set_ret_code(-1);
        response.set_result(training::METRIC_BATCH_RESULT_REJECTED_INVALID);
        response.set_message(build_error);
        return;
    }
    response.set_ret_code(0);
    response.set_result(training::METRIC_BATCH_RESULT_DELIVERED);
    *response.mutable_batch() = *pending_batch_;
    FillAvailability(response);
}

void MetricEventJournal::Ack(
    const training::AckMetricBatchReq& request,
    training::AckMetricBatchRsp& response) {
    std::lock_guard<std::mutex> lock(mutex_);
    response.Clear();
    FillAvailability(response);
    *response.mutable_committed_cursor() = committed_cursor_;
    if (!ValidContract(request.contract()) ||
        !ValidConsumer(request.consumer()) ||
        !SameConsumer(request.consumer())) {
        response.set_ret_code(-1);
        response.set_result(
            training::METRIC_BATCH_ACK_RESULT_REJECTED_IDENTITY);
        response.set_message("metric ACK contract or consumer is invalid");
        return;
    }
    const auto& cursor = request.cursor();
    if (cursor.SerializeAsString() == committed_cursor_.SerializeAsString()) {
        response.set_ret_code(0);
        response.set_result(
            training::METRIC_BATCH_ACK_RESULT_ALREADY_APPLIED);
        response.set_message("metric batch was already acknowledged");
        return;
    }
    if (!pending_batch_ || !SameIdentity(cursor.source(), source_) ||
        cursor.acknowledged_batch_sequence() !=
            pending_batch_->batch_sequence() ||
        !SameDigest(cursor.acknowledged_batch_digest(),
                    pending_batch_->batch_digest())) {
        response.set_ret_code(-1);
        response.set_result(
            training::METRIC_BATCH_ACK_RESULT_REJECTED_CURSOR);
        response.set_message("metric ACK does not identify the pending batch");
        return;
    }
    uint64_t expected_event =
        committed_cursor_.acknowledged_event_sequence();
    if (pending_batch_->events_size() > 0 || pending_batch_->has_gap()) {
        expected_event = pending_batch_->last_event_sequence();
    }
    if (cursor.acknowledged_event_sequence() != expected_event) {
        response.set_ret_code(-1);
        response.set_result(
            training::METRIC_BATCH_ACK_RESULT_REJECTED_CURSOR);
        response.set_message("metric ACK event cursor is invalid");
        return;
    }
    const bool acknowledged_final_batch = pending_batch_->source_final();
    committed_cursor_ = cursor;
    while (!events_.empty() &&
           events_.front().event_sequence() <=
               committed_cursor_.acknowledged_event_sequence()) {
        event_bytes_ -= events_.front().ByteSizeLong();
        events_.pop_front();
        event_enqueued_at_.pop_front();
    }
    pending_batch_.reset();
    if (acknowledged_final_batch) final_batch_acknowledged_ = true;
    changed_.notify_all();
    response.set_ret_code(0);
    response.set_result(training::METRIC_BATCH_ACK_RESULT_APPLIED);
    response.set_message("metric batch acknowledged");
    *response.mutable_committed_cursor() = committed_cursor_;
    FillAvailability(response);
}

EpisodeMetricsWindow::EpisodeMetricsWindow(std::size_t capacity)
    : capacity_(capacity) {}

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
            aggregate.reward_transitions +=
                static_cast<uint64_t>(agent.transition_count);
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
                aggregate.reward_component_sums[item.first] += item.second;
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
        const auto latest_value =
            latest_training_episode.reward_component_sums.find(name);

        const std::string prefix =
            "server.training.reward.component." + name;
        AddDescriptor(snapshot, prefix + ".episode_mean.v1", name,
                      "reward_components", "episode_reward",
                      "reward/agent_episode", "mean",
                      training::METRIC_VALUE_KIND_MEAN,
                      training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
        if (aggregate_value !=
            training_aggregate.reward_component_sums.end()) {
            AddMean(snapshot, prefix + ".episode_mean.v1",
                    aggregate_value->second,
                    training_aggregate.completed_agents,
                    timestamp_unix_ms);
        }
        AddDescriptor(snapshot, prefix + ".transition_mean.v1", name,
                      "reward_components", "transition_reward",
                      "reward/transition", "mean",
                      training::METRIC_VALUE_KIND_MEAN,
                      training::METRIC_AGGREGATION_KIND_WEIGHTED_MEAN);
        if (aggregate_value !=
            training_aggregate.reward_component_sums.end()) {
            AddMean(snapshot, prefix + ".transition_mean.v1",
                    aggregate_value->second,
                    training_aggregate.reward_transitions,
                    timestamp_unix_ms);
        }
        AddDescriptor(snapshot, prefix + ".latest_episode_mean.v1", name,
                      "reward_components", "episode_reward",
                      "reward/agent_episode", "latest_mean",
                      training::METRIC_VALUE_KIND_MEAN,
                      training::METRIC_AGGREGATION_KIND_LATEST,
                      training::METRIC_WINDOW_KIND_INSTANT);
        if (has_latest_training_episode &&
            latest_value !=
                latest_training_episode.reward_component_sums.end()) {
            AddMean(snapshot, prefix + ".latest_episode_mean.v1",
                    latest_value->second,
                    latest_training_episode.completed_agents,
                    timestamp_unix_ms);
        }
    }
}
