#pragma once

#include "task/sample/rollout_types.h"
#include "task/protocol/training_namespaces.h"

#include <string>
#include <vector>

// Computes unnormalised backward GAE and value targets over one contiguous,
// single-Agent, single-pinned-model segment. The caller owns segment identity,
// close reason and terminal/bootstrap provenance.
bool EstimateRolloutSegment(
    const std::vector<RawRolloutTransition>& segment,
    double gamma,
    double gae_lambda,
    double final_next_value,
    std::vector<float>& advantages,
    std::vector<float>& value_targets,
    std::string& error);

// Projects one estimator result into the ProcessedTransition wire payload
// consumed by SamplePool and Learner. Segment lifecycle and envelope transport
// remain owned by the task runtime and SampleDistributor respectively.
bool ProjectProcessedSegment(
    const std::vector<RawRolloutTransition>& raw_segment,
    const std::vector<float>& advantages,
    const std::vector<float>& value_targets,
    const std::string& segment_id,
    const training::ModelIdentity& behavior_model,
    int observation_dimension,
    int action_count,
    const std::string& action_mask_mode,
    std::vector<training::ProcessedTransition>& processed,
    std::string& error);
