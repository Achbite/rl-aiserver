#pragma once
#include "proto/metrics/catalog.grpc.pb.h"
#include <functional>
#include <string>
#include <utility>
namespace rl_sdk {
namespace metric_wire = rl::training::v1;
class MetricCatalogService final : public metric_wire::MetricCatalogService::Service {
public:
    explicit MetricCatalogService(std::function<metric_wire::GetMetricCatalogRsp()> snapshot)
        : snapshot_(std::move(snapshot)) {}
    grpc::Status GetMetricCatalog(grpc::ServerContext*, const metric_wire::GetMetricCatalogReq*,
                                 metric_wire::GetMetricCatalogRsp* response) override {
        *response = snapshot_();
        return grpc::Status::OK;
    }
private:
    std::function<metric_wire::GetMetricCatalogRsp()> snapshot_;
};
inline void AddStatusMetric(metric_wire::GetMetricCatalogRsp& catalog,
                            const std::string& id, const std::string& label,
                            const std::string& category, const std::string& unit,
                            const std::string& scope, const std::string& method,
                            const std::string& field,
                            rl::training::v1::MetricValueType type = rl::training::v1::METRIC_VALUE_TYPE_UNSIGNED,
                            const std::string& count_field = {}) {
    auto* entry = catalog.add_entries();
    auto* definition = entry->mutable_definition();
    definition->set_metric_id(id);
    definition->set_display_name(label);
    definition->set_category(category);
    definition->set_unit(unit);
    definition->set_scope(scope);
    definition->set_value_type(type);
    definition->set_aggregation(type == metric_wire::METRIC_VALUE_TYPE_SUM_COUNT
        ? metric_wire::METRIC_AGGREGATION_MEAN : metric_wire::METRIC_AGGREGATION_LATEST);
    if (type == metric_wire::METRIC_VALUE_TYPE_SUM_COUNT) definition->set_denominator(scope);
    entry->set_status_method(method);
    entry->set_status_field(field);
    entry->set_status_count_field(count_field);
}
}  // namespace rl_sdk
