#include "sample/sample_sender.h"

#include "log/logger.h"

#include <algorithm>
#include <chrono>

namespace {

double ElapsedMs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

}  // namespace

SampleSender::SampleSender(SampleOutputConfig config)
    : config_(std::move(config)) {}

SampleSender::~SampleSender() {
    StopAndDrain();
}

bool SampleSender::ProbeDistributor() {
    maze::DistributorStatusReq request;
    maze::DistributorStatusRsp response;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::milliseconds(config_.health_timeout_ms));
    grpc::Status status = stub_->GetStatus(&context, request, &response);
    if (!status.ok()) {
        MarkDegraded(
            "sample ingress status failed: " + status.error_message());
        return false;
    }
    if (!response.ready() || !response.ingress_ready()) {
        MarkDegraded("sample ingress is not ready");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    distributor_instance_id_ = response.distributor_instance_id();
    ready_ = true;
    degraded_ = false;
    last_error_.clear();
    return true;
}

bool SampleSender::Start() {
    if (!config_.enabled) {
        MarkDegraded("sample output is disabled");
        return false;
    }

    std::string target = config_.host + ":" + std::to_string(config_.port);
    channel_ = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
    stub_ = maze::SampleDistributorService::NewStub(channel_);
    auto deadline = std::chrono::system_clock::now() +
                    std::chrono::milliseconds(config_.health_timeout_ms);
    if (!channel_->WaitForConnected(deadline)) {
        MarkDegraded("sample ingress connection timeout: " + target);
        return false;
    }
    if (!ProbeDistributor()) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        accepting_ = true;
        stop_requested_ = false;
        force_stop_ = false;
    }
    sender_thread_ = std::thread(&SampleSender::SenderLoop, this);
    LOG_INFO("SampleSender", "就绪: target=%s, distributor_instance_id=%s",
             target.c_str(), distributor_instance_id_.c_str());
    return true;
}

bool SampleSender::Enqueue(const maze::SampleBatch& batch) {
    QueueItem item;
    item.batch = batch;
    item.samples = batch.samples_size();
    item.estimated_bytes = static_cast<int64_t>(batch.ByteSizeLong());

    std::unique_lock<std::mutex> lock(mutex_);
    auto has_capacity = [this, &item]() {
        return force_stop_ || !accepting_ || degraded_ ||
               (queue_.size() < config_.outbound_max_fragments &&
                queue_estimated_bytes_ + item.estimated_bytes <=
                    static_cast<int64_t>(config_.outbound_max_estimated_bytes));
    };
    if (!space_cv_.wait_for(lock,
                            std::chrono::milliseconds(config_.enqueue_timeout_ms),
                            has_capacity)) {
        degraded_ = true;
        last_error_ = "outbound queue enqueue timeout";
        return false;
    }
    if (!accepting_ || force_stop_ || degraded_) {
        return false;
    }

    queue_.push_back(std::move(item));
    queue_samples_ += queue_.back().samples;
    queue_estimated_bytes_ += queue_.back().estimated_bytes;
    queue_high_watermark_ =
        std::max(queue_high_watermark_, static_cast<int64_t>(queue_.size()));
    queue_cv_.notify_one();
    return true;
}

bool SampleSender::SendFront(const QueueItem& item,
                             bool& duplicate,
                             int& attempts_used,
                             std::string& error) {
    duplicate = false;
    attempts_used = 0;
    const int attempts = std::max(1, config_.max_attempts);

    for (int attempt = 0; attempt < attempts; ++attempt) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (force_stop_) {
                error = "sender forced to stop";
                return false;
            }
            ++push_attempt_count_;
            ++attempts_used;
            if (item.attempts + attempt > 0) ++retry_attempt_count_;
        }

        maze::PushSamplesRsp response;
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() +
                             std::chrono::milliseconds(config_.rpc_timeout_ms));
        {
            std::lock_guard<std::mutex> rpc_lock(rpc_mutex_);
            active_rpc_ = &context;
        }

        auto start = std::chrono::steady_clock::now();
        grpc::Status status = stub_->PushSamples(&context, item.batch, &response);
        double latency_ms = ElapsedMs(start);
        {
            std::lock_guard<std::mutex> rpc_lock(rpc_mutex_);
            active_rpc_ = nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++push_rpc_count_;
            push_rpc_latency_sum_ms_ += latency_ms;
            push_rpc_latency_max_ms_ = std::max(push_rpc_latency_max_ms_, latency_ms);
        }

        if (status.ok() &&
            (response.result() == maze::PUSH_RESULT_ACCEPTED ||
             response.result() == maze::PUSH_RESULT_DUPLICATE)) {
            duplicate = response.result() == maze::PUSH_RESULT_DUPLICATE;
            return true;
        }

        if (status.ok()) {
            error = response.message();
            std::lock_guard<std::mutex> lock(mutex_);
            ++rejected_push_attempt_count_;
            if (response.result() == maze::PUSH_RESULT_REJECTED_INVALID) {
                return false;
            }
        } else {
            error = status.error_message();
        }

        if (attempt + 1 < attempts) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(50 * (attempt + 1)));
        }
    }
    return false;
}

void SampleSender::SenderLoop() {
    while (true) {
        QueueItem item;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            queue_cv_.wait(lock, [this]() {
                return force_stop_ || !queue_.empty() || stop_requested_;
            });
            if (force_stop_ || (stop_requested_ && queue_.empty())) {
                break;
            }
            item = queue_.front();
        }

        bool duplicate = false;
        int attempts_used = 0;
        std::string error;
        if (SendFront(item, duplicate, attempts_used, error)) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!queue_.empty() &&
                queue_.front().batch.batch_id() == item.batch.batch_id()) {
                queue_samples_ -= queue_.front().samples;
                queue_estimated_bytes_ -= queue_.front().estimated_bytes;
                queue_.pop_front();
                accepted_unique_samples_ += item.samples;
                ++accepted_unique_batches_;
                if (duplicate) ++duplicate_push_attempt_count_;
                space_cv_.notify_all();
                if (queue_.empty()) drained_cv_.notify_all();
            }
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!queue_.empty() &&
                queue_.front().batch.batch_id() == item.batch.batch_id()) {
                queue_.front().attempts += attempts_used;
            }
        }
        MarkDegraded(error.empty() ? "PushSamples failed" : error);
        std::unique_lock<std::mutex> lock(mutex_);
        queue_cv_.wait_for(lock, std::chrono::seconds(1), [this]() {
            return force_stop_;
        });
    }

    std::lock_guard<std::mutex> lock(mutex_);
    ready_ = false;
    drained_cv_.notify_all();
}

void SampleSender::CancelActiveRpc() {
    std::lock_guard<std::mutex> lock(rpc_mutex_);
    if (active_rpc_) {
        active_rpc_->TryCancel();
    }
}

bool SampleSender::StopAndDrain() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!sender_thread_.joinable()) {
            accepting_ = false;
            ready_ = false;
            return queue_.empty();
        }
        accepting_ = false;
        stop_requested_ = true;
        queue_cv_.notify_all();
    }

    bool drained = false;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        drained = drained_cv_.wait_for(
            lock, std::chrono::milliseconds(config_.drain_timeout_ms),
            [this]() { return queue_.empty(); });
        if (!drained) {
            for (const auto& item : queue_) {
                final_drop_unique_samples_ += item.samples;
                ++final_drop_unique_batches_;
            }
            queue_.clear();
            queue_samples_ = 0;
            queue_estimated_bytes_ = 0;
            degraded_ = true;
            last_error_ = "outbound drain deadline exceeded";
            force_stop_ = true;
            queue_cv_.notify_all();
            space_cv_.notify_all();
        }
    }
    if (!drained) CancelActiveRpc();
    if (sender_thread_.joinable()) sender_thread_.join();
    return drained;
}

bool SampleSender::IsReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ready_;
}

bool SampleSender::IsDegraded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return degraded_;
}

void SampleSender::MarkDegraded(const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    degraded_ = true;
    last_error_ = error;
    space_cv_.notify_all();
}

void SampleSender::RecordFinalDrop(int64_t samples, int64_t batches,
                                   const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    final_drop_unique_samples_ += samples;
    final_drop_unique_batches_ += batches;
    degraded_ = true;
    last_error_ = error;
}

SampleSender::Snapshot SampleSender::GetSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Snapshot snapshot;
    snapshot.ready = ready_;
    snapshot.degraded = degraded_;
    snapshot.queue_fragments = queue_.size();
    snapshot.queue_samples = queue_samples_;
    snapshot.queue_estimated_bytes = queue_estimated_bytes_;
    snapshot.queue_high_watermark = queue_high_watermark_;
    snapshot.push_attempt_count = push_attempt_count_;
    snapshot.accepted_unique_samples = accepted_unique_samples_;
    snapshot.accepted_unique_batches = accepted_unique_batches_;
    snapshot.duplicate_push_attempt_count = duplicate_push_attempt_count_;
    snapshot.rejected_push_attempt_count = rejected_push_attempt_count_;
    snapshot.retry_attempt_count = retry_attempt_count_;
    snapshot.final_drop_unique_samples = final_drop_unique_samples_;
    snapshot.final_drop_unique_batches = final_drop_unique_batches_;
    snapshot.push_rpc_count = push_rpc_count_;
    snapshot.push_rpc_latency_sum_ms = push_rpc_latency_sum_ms_;
    snapshot.push_rpc_latency_max_ms = push_rpc_latency_max_ms_;
    snapshot.distributor_instance_id = distributor_instance_id_;
    snapshot.last_error = last_error_;
    return snapshot;
}
