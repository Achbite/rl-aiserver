#include "contracts/sample_producer_identity.h"

#include <iostream>

int main() {
    rl::common::v1::ServiceInstanceIdentity identity;
    aiserver_contract::FillSampleProducerIdentity(
        "aiserver-instance-1", 7, &identity);
    if (identity.component() != "aiserver" ||
        identity.instance_id() != "aiserver-instance-1" ||
        identity.lifecycle_epoch() != 7) {
        std::cerr << "AIServer sample producer identity is not canonical\n";
        return 1;
    }
    return 0;
}
