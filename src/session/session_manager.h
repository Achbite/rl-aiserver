#pragma once

#include "ai/astar_solver.h"
#include "ai/onnx_inferencer.h"
#include "contracts/contract_namespaces.h"
#include "model/behavior_policy_scope.h"
#include "model/model_manifest.h"
#include "model/model_step.h"
#include "session/lifecycle_replay_window.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
#include <optional>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>

// ---- 会话管理器（并行 Episode 隔离）----
// 每个 session 维护独立的 Agent 运行时状态和样本缓存，
// 不同 session 之间互不干扰，支持多 Episode 并行采集。

class SessionManager {
public:
    // One complete, trusted Environment transition before the AIServer closes
    // the Agent segment and computes GAE/value targets. Pending actions are not
    // represented here because they have no trusted result or next state yet.
    struct RawRolloutTransition {
        std::vector<float> observation;
        std::vector<float> next_observation;
        int action = 0;
        float reward = 0.0f;
        float behavior_log_probability = 0.0f;
        float behavior_value = 0.0f;
        uint64_t action_step = 0;
        int64_t created_at_unix_ms = 0;
    };

    // ---- Agent 运行时状态（每个 session 内独立）----
    struct AgentRuntime {
        AStarSolver solver;             // 独立寻路器
        int         last_action = 0;    // 上一帧动作
        bool        path_valid  = false;

        // 训练模式：帧样本缓存
        int   prev_grid_x  = -1;       // 待结算动作对应的前一环境状态
        int   prev_grid_y  = -1;
        bool  reached_goal = false;     // 本 Episode 是否到达终点
        bool  done_collected = false;   // 终止帧样本是否已收集（防止重复收集）
        bool  has_pending_action = false;
        int   pending_action = 0;
        int64_t pending_action_frame_id = -1;
        float pending_log_prob = 0.0f;
        float pending_value = 0.0f;
        std::vector<float> pending_obs;

        // R-PIN segment state. The prepared ORT session is copied as a shared
        // owner so model-cache pruning cannot invalidate in-flight inference.
        bool segment_open = false;
        std::string segment_id;
        ModelManifest pinned_model;
        OnnxInferencer::PreparedModel pinned_prepared_model;
        std::vector<RawRolloutTransition> segment_transitions;
        bool activated_model_seen = false;
        ModelStep last_activated_model_step = 0;
        std::string last_activated_model_lineage_id;
        int64_t last_completed_transition_at_unix_ms = 0;

        // --- 奖励与 observation 辅助状态 ---
        std::unordered_set<int> visited;
        bool current_state_first_visit = false;
        float first_visit_bonus_total = 0.0f;
        int episode_start_geodesic_distance = -1;
        int observation_grid_x = -1;
        int observation_grid_y = -1;
        bool last_move_blocked = false;
        int64_t blocked_move_count = 0;
        bool observation_done = false;
        int64_t last_observation_frame_id = -1;
        int64_t terminal_frame_id = -1;
        double episode_return = 0.0;
        int64_t episode_transition_count = 0;
        bool episode_behavior_model_seen = false;
        uint64_t minimum_episode_behavior_model_step = 0;
        uint64_t maximum_episode_behavior_model_step = 0;
        std::string episode_behavior_model_lineage_id;
        maze::MazeTerminationReason final_termination_reason =
            maze::MAZE_TERMINATION_REASON_UNSPECIFIED;
        std::unordered_map<std::string, double> reward_component_sums;
    };

    // ---- 单个会话 ----
    struct Session {
        std::string session_id;
        common::ServiceInstanceIdentity client;
        std::string environment_instance_id;
        int64_t last_valid_client_activity_unix_ms = 0;
        uint64_t session_epoch = 0;
        uint64_t last_command_sequence = 0;
        maze::SessionPhase phase = maze::SESSION_PHASE_OPEN;
        LifecycleReplayWindow command_replay;
        std::string map_id;
        std::string map_checksum_sha256;
        int shortest_action_steps = 0;
        std::string action_rule_id;
        maze::WorkloadMode workload_mode =
            maze::WORKLOAD_MODE_UNSPECIFIED;
        std::unordered_map<int, AgentRuntime> agents;   // agent_id → 运行时状态
        std::string current_episode_id;
        int64_t last_frame_id = -1;
        std::vector<maze::AgentAction> last_actions;
        // 地图参数（每个 session 独立，支持不同地图配置）
        float map_width  = 0.0f;
        float map_height = 0.0f;
        float start_x    = 0.0f;
        float start_y    = 0.0f;
        float end_x      = 0.0f;
        float end_y      = 0.0f;

        // 网格参数（Init 时计算）
        int start_gx  = 0;
        int start_gy  = 0;
        int end_gx    = 0;              // 终点网格坐标
        int end_gy    = 0;
        int grid_cols = 0;              // 网格列数
        int grid_rows = 0;              // 网格行数
        uint32_t grid_size_microunits = 0;

        // --- AIServer 校验后的 authoritative 网格与 geodesic 距离 ---
        std::vector<bool> blocked;
        std::vector<int> geodesic_distance;
        int max_finite_geodesic_distance = -1;
        maze::EpisodeMode current_episode_mode =
            maze::EPISODE_MODE_UNSPECIFIED;
        BehaviorPolicyScope behavior_policy_scope =
            BehaviorPolicyScope::Unspecified;
        int current_max_steps = 0;
        std::string evaluation_pinned_model_checksum;

        // 网格是否可通行（越界视为不可通行）
        bool IsWalkable(int gx, int gy) const {
            if (gx < 0 || gx >= grid_cols || gy < 0 || gy >= grid_rows) return false;
            return !blocked[gy * grid_cols + gx];
        }

    };

    struct ClientActivitySnapshot {
        int active_session_count = 0;
        int recent_active_session_count = 0;
        int64_t latest_active_activity_unix_ms = 0;
    };

    SessionManager() = default;
    ~SessionManager() = default;

    // Session ID 只由 AIServer 分配。
    std::string CreateSession();

    // 获取指定会话（不存在则返回 nullptr）
    Session* GetSession(const std::string& session_id);

    // 销毁指定会话
    void DestroySession(const std::string& session_id);

    // 获取当前活跃会话数
    int GetActiveSessionCount() const;
    ClientActivitySnapshot GetClientActivitySnapshot(
        int64_t now_unix_ms, int64_t lease_ms) const;
    int GetActiveEpisodeCount() const;
    std::vector<std::string> GetSessionIds() const;

private:
    std::unordered_map<std::string, Session> sessions_;
    mutable std::mutex mutex_;
    uint64_t next_session_id_ = 1;
};
