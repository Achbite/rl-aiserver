#pragma once

#include "ai/onnx_inferencer.h"
#include "contracts/training_namespaces.h"
#include "metrics/reward_metrics.h"
#include "model/behavior_policy_scope.h"
#include "model/model_manifest.h"
#include "model/model_step.h"
#include "policy/episode_mode.h"
#include "proto/communication/session.pb.h"
#include "rl_sdk/replay_window.h"
#include "sample/rollout_types.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct AgentTrainingState {
    bool done_collected = false;
    bool has_pending_action = false;
    int pending_action = 0;
    int64_t pending_action_frame_id = -1;
    float pending_log_prob = 0.0f;
    float pending_value = 0.0f;
    std::vector<float> pending_obs;
    std::vector<bool> pending_action_mask;

    // The shared prepared model keeps in-flight segments valid during pruning.
    bool segment_open = false;
    std::string segment_id;
    ModelManifest pinned_model;
    OnnxInferencer::PreparedModel pinned_prepared_model;
    std::vector<RawRolloutTransition> segment_transitions;
    bool activated_model_seen = false;
    ModelStep last_activated_model_step = 0;
    std::string last_activated_model_lineage_id;
    int64_t last_completed_transition_at_unix_ms = 0;

    bool observation_done = false;
    int64_t last_observation_frame_id = -1;
    int64_t terminal_frame_id = -1;
    double episode_return = 0.0;
    int64_t episode_transition_count = 0;
    bool episode_behavior_model_seen = false;
    uint64_t minimum_episode_behavior_model_step = 0;
    uint64_t maximum_episode_behavior_model_step = 0;
    std::string episode_behavior_model_lineage_id;
    std::unordered_map<std::string, double> reward_component_sums;

    void ResetForEpisode() {
        done_collected = false;
        has_pending_action = false;
        pending_action = 0;
        pending_action_frame_id = -1;
        pending_log_prob = 0.0f;
        pending_value = 0.0f;
        pending_obs.clear();
        pending_action_mask.clear();
        segment_open = false;
        segment_id.clear();
        pinned_model = ModelManifest{};
        pinned_prepared_model = OnnxInferencer::PreparedModel{};
        segment_transitions.clear();
        last_completed_transition_at_unix_ms = 0;
        observation_done = false;
        last_observation_frame_id = -1;
        terminal_frame_id = -1;
        episode_return = 0.0;
        episode_transition_count = 0;
        episode_behavior_model_seen = false;
        minimum_episode_behavior_model_step = 0;
        maximum_episode_behavior_model_step = 0;
        episode_behavior_model_lineage_id.clear();
        reward_component_sums.clear();
        // Model activation history belongs to the Agent lifecycle across episodes.
    }
};

template <class AgentState>
struct TrainingSession {
    using AgentRuntime = AgentState;

    RewardMetricWindow reward_metrics;
    std::string session_id;
    common::ServiceInstanceIdentity client;
    std::string environment_instance_id;
    int64_t last_valid_client_activity_unix_ms = 0;
    uint64_t session_epoch = 0;
    uint64_t last_command_sequence = 0;
    rl::session::v1::SessionPhase phase = rl::session::v1::SESSION_PHASE_OPEN;
    LifecycleReplayWindow command_replay;
    PolicyMode workload_mode = PolicyMode::Unspecified;
    std::unordered_map<int, AgentState> agents;
    std::string current_episode_id;
    int64_t last_frame_id = -1;
    PolicyMode current_episode_mode = PolicyMode::Unspecified;
    BehaviorPolicyScope behavior_policy_scope = BehaviorPolicyScope::Unspecified;
    std::string evaluation_pinned_model_lineage_id;
    ModelStep evaluation_pinned_model_step = 0;

    void ResetForEpisode(const std::string& episode_id) {
        current_episode_id = episode_id;
        last_frame_id = -1;
        for (auto& item : agents) {
            item.second.ResetForEpisode();
        }
    }
};
