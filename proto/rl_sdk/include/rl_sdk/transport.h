#pragma once
#include <grpcpp/grpcpp.h>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

namespace rl_sdk {
class Transport {
public:
    bool Connect(const std::string& target) {
        channel_ = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
        connected_ = channel_->WaitForConnected(std::chrono::system_clock::now() + std::chrono::seconds(5));
        return connected_;
    }
    void Disconnect() { connected_ = false; channel_.reset(); }
    bool connected() const { return connected_; }
    std::shared_ptr<grpc::Channel> channel() const { return channel_; }
    bool outcome_unknown() const { return outcome_unknown_; }
    const std::string& error() const { return error_; }
    template<class Response, class Invoke>
    bool InvokeUnary(Response& response, std::chrono::seconds timeout, Invoke invoke) {
        outcome_unknown_ = false;
        error_.clear();
        if (!connected_) { error_ = "transport is not connected"; return false; }
        for (int attempt = 1; attempt <= 2; ++attempt) {
            grpc::ClientContext context;
            context.set_deadline(std::chrono::system_clock::now() + timeout);
            Response candidate;
            const auto status = invoke(context, candidate);
            if (status.ok()) { outcome_unknown_ = false; error_.clear(); response.Swap(&candidate); return true; }
            outcome_unknown_ = true;
            error_ = "gRPC status " + std::to_string(static_cast<int>(status.error_code())) +
                ": " + status.error_message();
            const auto code = status.error_code();
            const bool retryable = code == grpc::StatusCode::ABORTED ||
                code == grpc::StatusCode::CANCELLED || code == grpc::StatusCode::DEADLINE_EXCEEDED ||
                code == grpc::StatusCode::INTERNAL || code == grpc::StatusCode::RESOURCE_EXHAUSTED ||
                code == grpc::StatusCode::UNKNOWN || code == grpc::StatusCode::UNAVAILABLE;
            if (!retryable || attempt == 2) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return false;
    }
private:
    std::shared_ptr<grpc::Channel> channel_;
    bool connected_ = false;
    bool outcome_unknown_ = false;
    std::string error_;
};
}  // namespace rl_sdk
