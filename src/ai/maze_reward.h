#pragma once

#include "session/session_manager.h"

#include <vector>
#include <string>
#include <utility>

// ---- 奖励计算结果（含分项明细）----
struct RewardDetail {
    bool valid = true;
    std::string error;
    float total = 0.0f;                                 // 总奖励
    float task_total = 0.0f;                            // Goal/Timeout
    float shaping_total = 0.0f;                         // Geodesic/First-Visit
    std::vector<std::pair<std::string, float>> items;   // 分项明细：<奖励名, 值>
};

struct MazeRewardConfig {
    float goal_reward = 10.0f;
    float timeout_penalty = -2.0f;
    float progress_budget = 1.0f;
    float stage_8x_first_visit_budget = 0.75f;
    float stage_4x_first_visit_budget = 0.25f;
    float stage_2x_first_visit_budget = 0.0f;
    float wasted_action_penalty = -0.002f;
};

// ---- 迷宫奖励计算器 ----
// 独立模块，负责所有奖励函数的计算和分项记录。
// Calculate() 返回的 items 是 Dashboard 奖励分项指标的数据源。
class MazeReward {
public:
    // 计算单帧总奖励（含分项明细）
    static RewardDetail Calculate(const SessionManager::Session& session,
                                  int agent_id, int gx, int gy, bool is_done,
                                  maze::MazeTerminationReason reason,
                                  const MazeRewardConfig& config);
};
