#pragma once

#include "common.pb.h"

#include <cstdint>
#include <string>

namespace aiserver_contract {

inline constexpr char kSampleProducerComponent[] = "aiserver";

inline void FillSampleProducerIdentity(
    const std::string& instance_id,
    std::uint64_t lifecycle_epoch,
    rl::common::v1::ServiceInstanceIdentity* target) {
    target->set_component(kSampleProducerComponent);
    target->set_instance_id(instance_id);
    target->set_lifecycle_epoch(lifecycle_epoch);
}

}  // namespace aiserver_contract
