#pragma once
#include "task/metrics/episode_metrics.h"
#include "proto/training/training.grpc.pb.h"
#include "proto/metrics/transport.grpc.pb.h"

class MetricEventService final : public training::MetricEventService::Service {
public:
    explicit MetricEventService(MetricEventJournal& journal) : journal_(journal) {}
    grpc::Status GetMetricBatch(grpc::ServerContext*, const training::GetMetricBatchReq* request,
                               training::GetMetricBatchRsp* response) override {
        journal_.Get(*request, *response);
        return grpc::Status::OK;
    }
    grpc::Status AckMetricBatch(grpc::ServerContext*, const training::AckMetricBatchReq* request,
                               training::AckMetricBatchRsp* response) override {
        journal_.Ack(*request, *response);
        return grpc::Status::OK;
    }
private:
    MetricEventJournal& journal_;
};
