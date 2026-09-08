#pragma once

#include "contracts/contract_namespaces.h"
#include "metrics/metric_registry.h"
#include "sample/reward_result.h"

#include <vector>
#include <string>
#include <utility>

// ---- 奖励计算结果（含分项明细）----
struct MazeRewardDetail : RewardResult {
    float task_total = 0.0f;                            // Goal/Timeout
    float shaping_total = 0.0f;                         // Geodesic/First-Visit
};

struct MazeRewardParameters {
    double goal_reward;
    double timeout_penalty;
    double progress_budget;
    double stage_8x_first_visit_budget;
    double stage_4x_first_visit_budget;
    double stage_2x_first_visit_budget;
    double wasted_action_penalty;
};

// Reward parameters are owned by the AIServer reward implementation.
const MazeRewardParameters& GetMazeRewardParameters();

struct MazeRewardContext {
    int grid_cols;
    int grid_rows;
    const std::vector<int>& geodesic_distance;
    int shortest_action_steps;
    int prev_grid_x;
    int prev_grid_y;
    int episode_start_geodesic_distance;
    bool current_state_first_visit;
    float first_visit_bonus_total;
};

// ---- 迷宫奖励计算器 ----
// 独立模块，负责所有奖励函数的计算和分项记录。
// Calculate() 返回的 items 是 Dashboard 奖励分项指标的数据源。
class MazeReward {
public:
    static void RegisterMetrics(MetricRegistry& registry);
    // 计算单帧总奖励（含分项明细）
    static MazeRewardDetail Calculate(const MazeRewardContext& context,
                                  int gx, int gy, bool is_done,
                                  maze::MazeTerminationReason reason);
};
