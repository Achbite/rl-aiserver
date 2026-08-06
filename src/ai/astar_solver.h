#pragma once

#include <vector>
#include <cstdint>

// ---- A* 网格坐标 ----
struct GridPos {
    int gx = 0;     // 网格 X 索引
    int gy = 0;     // 网格 Y 索引
};

// ---- A* 寻路器 ----
class AStarSolver {
public:
    // 使用 Client 回报且经 AIServer 校验的 authoritative bitmap 初始化。
    bool InitGrid(int grid_cols, int grid_rows,
                  const std::vector<bool>& blocked);

    // 使用真实动作网格规划 unit-cost 最短路径。
    bool PlanPath(int start_gx, int start_gy, int goal_gx, int goal_gy);

    // 根据当前网格坐标获取推荐动作 ID（0-8）
    int GetAction(int cur_gx, int cur_gy) const;

    // 获取路径点数量
    int GetPathLength() const { return static_cast<int>(path_.size()); }

    // 调试：获取路径点（连续坐标）
    const std::vector<GridPos>& GetPath() const { return path_; }

private:
    // 网格是否可通行
    bool IsWalkable(int gx, int gy) const;

    // 网格参数
    int   grid_cols_  = 0;          // 网格列数
    int   grid_rows_  = 0;          // 网格行数

    // 障碍物网格（true=不可通行）
    std::vector<bool> blocked_;

    // 规划结果（网格坐标序列，从起点到终点）
    std::vector<GridPos> path_;
};
