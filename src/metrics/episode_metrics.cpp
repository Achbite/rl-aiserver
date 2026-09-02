#include "metrics/episode_metrics.h"

#include <algorithm>
#include <utility>

namespace {

bool SameIdentity(const common::ServiceInstanceIdentity& left,
                  const common::ServiceInstanceIdentity& right) {
    return left.component() == right.component() &&
           left.instance_id() == right.instance_id() &&
           left.lifecycle_epoch() == right.lifecycle_epoch();
}

}  // namespace

MetricEventJournal::MetricEventJournal(
    common::ServiceInstanceIdentity source,
    std::size_t capacity,
    std::size_t byte_capacity,
    std::chrono::milliseconds flush_interval)
    : source_(std::move(source)),
      capacity_(capacity),
      byte_capacity_(byte_capacity),
      flush_interval_(flush_interval),
      last_batch_created_at_(std::chrono::steady_clock::now()) {
    *committed_cursor_.mutable_source() = source_;
}

MetricEventJournal::AppendResult MetricEventJournal::AppendFact(
    std::string fact_payload,
    int64_t observed_at_unix_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    AppendResult result;
    if (source_final_) {
        result.code = AppendResult::Code::SourceFinal;
        return result;
    }
    training::MetricEvent event;
    if (last_event_observed_at_unix_ms_ &&
        observed_at_unix_ms < *last_event_observed_at_unix_ms_) {
        result.wall_clock_regressed = true;
        result.previous_observed_at_unix_ms =
            *last_event_observed_at_unix_ms_;
    }
    event.set_event_sequence(next_event_sequence_++);
    event.set_observed_at_unix_ms(observed_at_unix_ms);
    event.set_fact_kind(training::METRIC_FACT_KIND_MAZE_EPISODE);
    event.set_fact_payload(std::move(fact_payload));
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
    return true;
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

bool MetricEventJournal::BuildPendingBatch(
    const training::GetMetricBatchReq& request,
    int64_t now_unix_ms,
    std::string& error) {
    training::MetricBatch batch;
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
    if (!ValidConsumer(request.consumer())) {
        response.set_result(training::METRIC_BATCH_RESULT_REJECTED_INVALID);
        response.set_message("metric consumer lifecycle identity is invalid");
        return;
    }
    if (consumer_ && !SameConsumer(request.consumer())) {
        response.set_result(training::METRIC_BATCH_RESULT_REJECTED_INVALID);
        response.set_message("metric journal is pinned to another consumer");
        return;
    }
    if (!CursorMatchesCommitted(request.cursor())) {
        response.set_result(
            training::METRIC_BATCH_RESULT_REJECTED_CURSOR);
        response.set_message("metric cursor does not match committed cursor");
        return;
    }
    if (request.max_events() == 0 || request.max_events() > 1024 ||
        request.max_bytes() <= 0 || request.max_bytes() > 16 * 1024 * 1024 ||
        request.wait_timeout_ms() < 0 || request.wait_timeout_ms() > 5000) {
        response.set_result(
            training::METRIC_BATCH_RESULT_REJECTED_INVALID);
        response.set_message("metric batch limits are invalid");
        return;
    }
    if (!consumer_) consumer_ = request.consumer();
    if (pending_batch_) {
        response.set_result(training::METRIC_BATCH_RESULT_DELIVERED);
        *response.mutable_batch() = *pending_batch_;
        FillAvailability(response);
        return;
    }
    if (source_final_ && final_batch_acknowledged_) {
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
        response.set_result(training::METRIC_BATCH_RESULT_WAIT);
        response.set_message("metric flush window has not closed");
        return;
    }
    const int64_t now_unix_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    std::string build_error;
    if (!BuildPendingBatch(request, now_unix_ms, build_error)) {
        response.set_result(training::METRIC_BATCH_RESULT_REJECTED_INVALID);
        response.set_message(build_error);
        return;
    }
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
    if (!ValidConsumer(request.consumer()) ||
        !SameConsumer(request.consumer())) {
        response.set_result(training::METRIC_BATCH_ACK_RESULT_REJECTED_INVALID);
        response.set_message("metric ACK consumer lifecycle is invalid");
        return;
    }
    const auto& cursor = request.cursor();
    if (cursor.SerializeAsString() == committed_cursor_.SerializeAsString()) {
        response.set_result(
            training::METRIC_BATCH_ACK_RESULT_ALREADY_APPLIED);
        response.set_message("metric batch was already acknowledged");
        return;
    }
    if (!pending_batch_ || !SameIdentity(cursor.source(), source_) ||
        cursor.acknowledged_batch_sequence() !=
            pending_batch_->batch_sequence()) {
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
    response.set_result(training::METRIC_BATCH_ACK_RESULT_APPLIED);
    response.set_message("metric batch acknowledged");
    *response.mutable_committed_cursor() = committed_cursor_;
    FillAvailability(response);
}
