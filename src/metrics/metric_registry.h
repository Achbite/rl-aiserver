#pragma once

#include "contracts/training_namespaces.h"

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

// One registry per producer lifecycle. Definitions travel with the points that
// use them; retention does not require replaying a separate registration event.
class MetricRegistry {
public:
    std::string Register(const std::string& id, const std::string& label,
                  const std::string& unit, const std::string& scope,
                  training::MetricValueType type,
                  training::MetricAggregation aggregation,
                  const std::string& denominator = {}) {
        const auto existing = definitions_.find(id);
        if (existing != definitions_.end()) {
            const auto& definition = existing->second;
            if (definition.display_name() != label || definition.unit() != unit ||
                definition.scope() != scope || definition.value_type() != type ||
                definition.aggregation() != aggregation || definition.denominator() != denominator) {
                throw std::invalid_argument("metric definition changed: " + id);
            }
            return id;
        }
        const bool mean = aggregation == training::METRIC_AGGREGATION_MEAN;
        if (id.empty() || label.empty() || unit.empty() || scope.empty() ||
            !training::MetricValueType_IsValid(type) || type == 0 ||
            !training::MetricAggregation_IsValid(aggregation) || aggregation == 0 ||
            mean != (type == training::METRIC_VALUE_TYPE_SUM_COUNT) ||
            mean != !denominator.empty()) {
            throw std::invalid_argument("invalid metric definition: " + id);
        }
        training::MetricDefinition definition;
        definition.set_metric_id(id);
        definition.set_display_name(label);
        definition.set_unit(unit);
        definition.set_scope(scope);
        definition.set_value_type(type);
        definition.set_aggregation(aggregation);
        definition.set_denominator(denominator);
        definitions_.emplace(id, std::move(definition));
        return id;
    }

    training::MetricPoint& Mean(training::RegisteredMetricRecord& record,
                               const std::string& id, double sum, uint64_t count) const {
        if (!std::isfinite(sum) || count == 0) {
            throw std::invalid_argument("invalid metric sum/count: " + id);
        }
        auto& point = Add(record, id, training::METRIC_VALUE_TYPE_SUM_COUNT);
        point.mutable_sum_count()->set_sum(sum);
        point.mutable_sum_count()->set_count(count);
        return point;
    }

    training::MetricPoint& Scalar(training::RegisteredMetricRecord& record,
                                 const std::string& id, double value) const {
        if (!std::isfinite(value)) throw std::invalid_argument("non-finite metric: " + id);
        auto& point = Add(record, id, training::METRIC_VALUE_TYPE_SCALAR);
        point.set_scalar(value);
        return point;
    }

    training::MetricPoint& Unsigned(training::RegisteredMetricRecord& record,
                                   const std::string& id, uint64_t value) const {
        auto& point = Add(record, id, training::METRIC_VALUE_TYPE_UNSIGNED);
        point.set_unsigned_value(value);
        return point;
    }

private:
    training::MetricPoint& Add(training::RegisteredMetricRecord& record,
                              const std::string& id, training::MetricValueType type) const {
        auto found = definitions_.find(id);
        if (found == definitions_.end() || found->second.value_type() != type) {
            throw std::invalid_argument("unregistered metric or wrong value type: " + id);
        }
        bool included = false;
        for (const auto& definition : record.definitions()) {
            if (definition.metric_id() == id) { included = true; break; }
        }
        if (!included) *record.add_definitions() = found->second;
        auto* point = record.add_points();
        point->set_metric_id(id);
        return *point;
    }
    std::map<std::string, training::MetricDefinition> definitions_;
};
