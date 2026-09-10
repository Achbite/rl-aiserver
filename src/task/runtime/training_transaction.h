#pragma once

#include "task/runtime/training_runtime.h"
#include "task/runtime/task_input.h"
#include <algorithm>
#include <cmath>

// A command's training state and outputs commit together after sample admission.
// The task contributes normalized inputs, without touching counters or queues.
template<class Sessions>
class TrainingTransaction {
public:
    using Runtime = TrainingRuntime<Sessions>;
    using Session = typename Sessions::Session;
    TrainingTransaction(Runtime& runtime, const Session& source)
        : session(source), runtime_(runtime), transitions(runtime.produced_unique_transitions_),
          envelopes(runtime.produced_unique_envelopes_), by_model(runtime.produced_transitions_by_model_),
          segment_sequence(runtime.next_segment_seq_.load()), activations(runtime.per_agent_model_activation_count_),
          latest_used(runtime.latest_prepared_used_by_agent_), closed(runtime.closed_segment_count_),
          quarantined(runtime.quarantined_transition_count_), excluded(runtime.pending_action_excluded_count_),
          close_counts(runtime.segment_close_counts_), rng(runtime.action_rng_) {}

    bool PrepareFrame(const std::vector<AgentTaskInput>& inputs, int64_t frame, bool collect,
                      std::vector<ModelTaskAction>& actions, std::string& error) {
        for (const auto& input : inputs) {
            if (!runtime_.FinalizePendingTransition(session, input.agent_id, input.reward,
                    input.observation, collect, error)) {
                ++runtime_.rollout_estimator_failure_count_;
                runtime_.MarkDegraded("Episode transition preparation failed: " + error);
                return false;
            }
            auto& agent = session.agents.at(input.agent_id);
            if (input.terminal) {
                agent.done_collected = true;
                agent.terminal_frame_id = frame;
            }
        }
        if (collect) {
            for (const auto& input : inputs) {
                auto& agent = session.agents.at(input.agent_id);
                if (!agent.segment_open || agent.segment_transitions.empty()) continue;
                auto reason = training::SEGMENT_CLOSE_REASON_ENVIRONMENT_TERMINATED;
                float bootstrap = 0.0f;
                if (!input.terminal) {
                    if (agent.segment_transitions.size() > runtime_.config_.rollout.tmax) {
                        error = "Agent segment exceeded its configured TMax";
                        ++runtime_.rollout_estimator_failure_count_;
                        runtime_.MarkDegraded(error);
                        return false;
                    }
                    if (agent.segment_transitions.size() != runtime_.config_.rollout.tmax) continue;
                    reason = training::SEGMENT_CLOSE_REASON_TMAX;
                    if (!runtime_.InferPinnedValue(agent, agent.segment_transitions.back().next_observation, bootstrap)) {
                        error = runtime_.last_error_;
                        ++runtime_.rollout_estimator_failure_count_;
                        return false;
                    }
                }
                if (!Close(input.agent_id, reason, bootstrap, !input.terminal, error)) return false;
            }
        }
        for (const auto& input : inputs) {
            if (input.terminal) continue;
            auto& agent = session.agents.at(input.agent_id);
            int action = 0;
            float log_probability = 0, value = 0;
            if (!runtime_.PrepareModelAction(session, agent, input.agent_id, input.observation,
                    frame, input.action_mask, segment_sequence, activations, latest_used, rng,
                    action, log_probability, value)) {
                error = runtime_.last_error_;
                return false;
            }
            actions.push_back({input.agent_id, action});
        }
        session.last_frame_id = frame;
        const bool all_done = std::all_of(session.agents.begin(), session.agents.end(),
            [](const auto& item) { return item.second.done_collected; });
        if (all_done) session.phase = rl::session::v1::SESSION_PHASE_EPISODE_TERMINAL;
        return true;
    }

    bool PrepareAbort(bool collect, std::string& error) {
        const auto reason = training::SEGMENT_CLOSE_REASON_CLIENT_CONTROLLED_CLOSE;
        for (auto& item : session.agents) {
            auto& agent = item.second;
            if (collect && (agent.segment_open || agent.has_pending_action || !agent.segment_transitions.empty())) {
                if (agent.segment_open && !agent.segment_transitions.empty()) {
                    const bool pending = agent.has_pending_action;
                    float bootstrap = pending ? agent.pending_value : 0.0f;
                    if ((pending && !std::isfinite(bootstrap)) ||
                        (!pending && !runtime_.InferPinnedValue(agent, agent.segment_transitions.back().next_observation, bootstrap))) {
                        ++runtime_.rollout_estimator_failure_count_;
                        error = pending ? "controlled close pending bootstrap is non-finite" : runtime_.last_error_;
                        runtime_.MarkDegraded(error);
                        return false;
                    }
                    if (!Close(item.first, reason, bootstrap, true, error)) return false;
                    if (pending) ++excluded;
                } else {
                    runtime_.DiscardAgentSegment(session, item.first, reason, true,
                        quarantined, excluded, closed, close_counts);
                }
            }
            agent.has_pending_action = false;
            agent.pending_action_frame_id = -1;
            agent.pending_obs.clear();
            agent.pending_action_mask.clear();
        }
        session.phase = rl::session::v1::SESSION_PHASE_ABORTED;
        ReleaseEpisodePolicy();
        return true;
    }

    SampleDistributor::ReservationResult Admit(std::string& error, bool record_latency) {
        using Result = SampleDistributor::ReservationResult;
        uint64_t reservation = 0;
        const auto start = std::chrono::steady_clock::now();
        const auto reserved = runtime_.sample_distributor_.ReserveEnqueueEnvelopeSet(output, reservation, error);
        if (reserved != Result::kReserved) return reserved;
        const auto sealed = runtime_.sample_distributor_.SealEnqueueEnvelopeSet(reservation, error);
        if (sealed == SampleDistributor::SealResult::kRetryableUnavailable) return Result::kRetryableUnavailable;
        if (sealed == SampleDistributor::SealResult::kTerminalFault) return Result::kTerminalFault;
        if (runtime_.sample_distributor_.CommitEnqueueEnvelopeSet(reservation, error) !=
                SampleDistributor::CommitResult::kCommitted) return Result::kTerminalFault;
        if (record_latency && !output.empty()) {
            const double elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            runtime_.enqueue_count_ += output.size();
            runtime_.enqueue_latency_sum_ms_ += elapsed;
            runtime_.enqueue_latency_max_ms_ = std::max(runtime_.enqueue_latency_max_ms_, elapsed);
        }
        return Result::kReserved;
    }

    void ReleaseEpisodePolicy() {
        session.behavior_policy_scope = BehaviorPolicyScope::Unspecified;
        session.evaluation_pinned_model_lineage_id.clear();
        session.evaluation_pinned_model_step = 0;
    }
    void Commit(Session& destination) {
        destination = std::move(session);
        runtime_.produced_unique_transitions_ = transitions;
        runtime_.produced_unique_envelopes_ = envelopes;
        runtime_.produced_transitions_by_model_ = std::move(by_model);
        runtime_.next_segment_seq_.store(segment_sequence);
        runtime_.per_agent_model_activation_count_ = activations;
        runtime_.latest_prepared_used_by_agent_ = latest_used;
        runtime_.closed_segment_count_ = closed;
        runtime_.quarantined_transition_count_ = quarantined;
        runtime_.pending_action_excluded_count_ = excluded;
        runtime_.segment_close_counts_ = std::move(close_counts);
        runtime_.action_rng_ = std::move(rng);
    }
    int64_t ProducedTransitions() const { return transitions; }
    Session session;

private:
    bool Close(int agent_id, training::SegmentCloseReason reason, float bootstrap,
               bool bootstrap_applied, std::string& error) {
        if (!runtime_.PrepareAgentSegmentClose(session, agent_id, reason, bootstrap,
                bootstrap_applied, transitions, envelopes, by_model, output, close_counts, error)) {
            ++runtime_.rollout_estimator_failure_count_;
            runtime_.MarkDegraded("Agent segment close failed: " + error);
            return false;
        }
        ++closed;
        return true;
    }
    Runtime& runtime_;
    int64_t transitions, envelopes;
    std::unordered_map<ModelStep, int64_t> by_model;
    uint64_t segment_sequence;
    int64_t activations;
    bool latest_used;
    int64_t closed, quarantined, excluded;
    std::unordered_map<training::SegmentCloseReason, int64_t> close_counts;
    std::mt19937 rng;
    std::vector<training::ProcessedTransitionEnvelope> output;
};
