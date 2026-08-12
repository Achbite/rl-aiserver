#pragma once

#include "ai/astar_solver.h"
#include "contracts/contract_namespaces.h"
#include "model/behavior_policy_scope.h"
#include "session/lifecycle_replay_window.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>

// ---- 会话管理器（并行 Episode 隔离）----
// 每个 session 维护独立的 Agent 运行时状态和样本缓存，
// 不同 session 之间互不干扰，支持多 Episode 并行采集。

class SessionManager {
public:
    enum class EpisodeState {
        None,
        Active,
        Ended,
        Aborted,
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
        int   pending_model_version = -1;
        std::string pending_model_checksum;
        std::string pending_model_lineage_id;
        std::string pending_model_manifest_digest;
        std::vector<float> pending_obs;
        int   fragment_model_version = -1;
        std::string fragment_model_checksum;
        std::string fragment_model_lineage_id;
        std::string fragment_model_manifest_digest;
        int64_t fragment_first_action_frame_id = -1;

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
        double episode_return = 0.0;
        int64_t episode_transition_count = 0;
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
        maze::TaskIdentity task;
        uint64_t lifecycle_epoch = 0;
        uint64_t last_command_sequence = 0;
        maze::TaskState task_state = maze::TASK_STATE_INITIALIZING;
        maze::SessionState session_state = maze::SESSION_STATE_OPENED;
        maze::EpisodeState protocol_episode_state =
            maze::EPISODE_STATE_UNSPECIFIED;
        maze::EvaluationState evaluation_state =
            maze::EVALUATION_STATE_INACTIVE;
        std::string current_evaluation_id;
        LifecycleReplayWindow command_replay;
        std::string map_id;
        std::string map_checksum_sha256;
        int shortest_action_steps = 0;
        std::string action_rule_id;
        maze::WorkloadMode workload_mode =
            maze::WORKLOAD_MODE_UNSPECIFIED;
        std::unordered_map<int, AgentRuntime> agents;   // agent_id → 运行时状态
        std::string current_episode_id;
        EpisodeState episode_state = EpisodeState::None;
        int64_t last_frame_id = -1;
        std::vector<maze::AgentAction> last_actions;
        std::unordered_map<int, std::vector<training::Sample>> agent_sample_caches;  // agent_id → 样本缓存（多 Agent 隔离）
        std::unordered_map<int, training::SampleBatch> pending_sample_batches;

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

        bool opened = false;
        bool initialized = false;       // 是否已初始化

        // --- AIServer 校验后的 authoritative 网格与 geodesic 距离 ---
        std::vector<bool> blocked;
        std::vector<int> geodesic_distance;
        int max_finite_geodesic_distance = -1;
        maze::CurriculumStage curriculum_stage =
            maze::CURRICULUM_STAGE_8X;
        maze::EpisodeMode current_episode_mode =
            maze::EPISODE_MODE_UNSPECIFIED;
        BehaviorPolicyScope behavior_policy_scope =
            BehaviorPolicyScope::Unspecified;
        int current_max_steps = 0;
    int evaluation_pinned_model_version = -1;
    std::string evaluation_pinned_model_checksum;
    std::string evaluation_pinned_model_lineage_id;
    std::string evaluation_pinned_model_manifest_digest;
    int64_t evaluation_pinned_model_train_updates = 0;
    int64_t evaluation_pinned_model_trained_samples = 0;

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
