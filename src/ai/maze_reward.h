#pragma once

#include "session/session_manager.h"

#include <vector>
#include <string>
#include <utility>

// ---- 奖励计算结果（含分项明细）----
struct RewardDetail {
    float total = 0.0f;                                 // 总奖励
    std::vector<std::pair<std::string, float>> items;   // 分项明细：<奖励名, 值>
};

// ---- 迷宫奖励计算器 ----
// 独立模块，负责所有奖励函数的计算和分项记录。
// Calculate() 返回的 items 是 Dashboard 奖励分项指标的数据源。
class MazeReward {
public:
    // 计算单帧总奖励（含分项明细）
    static RewardDetail Calculate(const SessionManager::Session& session,
                                  int agent_id, int gx, int gy, bool is_done,
                                  int agent_num);

    // 计算排名奖励（Episode 结束时调用）
    // ranking_order: Agent 完成排名顺序（先完成的在前）
    // agent_num: 总 Agent 数量
    // agent_id: 当前 Agent ID
    // reached_goal: 该 Agent 是否到达终点
    static float CalculateRankReward(const std::vector<int>& ranking_order,
                                     int agent_num, int agent_id, bool reached_goal);
};
