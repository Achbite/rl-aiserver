#include "maze/environment/map.h"

#include <algorithm>
#include <deque>
#include <limits>

namespace {

constexpr int kDirections[8][2] = {
    {0, 1}, {1, 1}, {1, 0}, {1, -1},
    {0, -1}, {-1, -1}, {-1, 0}, {-1, 1},
};

bool BasicShapeValid(const maze::MapDescriptor& descriptor,
                     std::string& error) {
    if (descriptor.grid_columns() == 0 || descriptor.grid_rows() == 0 ||
        descriptor.grid_size_microunits() == 0) {
        error = "unsupported map format or dimensions";
        return false;
    }
    const int64_t cell_count =
        static_cast<int64_t>(descriptor.grid_columns()) *
        static_cast<int64_t>(descriptor.grid_rows());
    if (cell_count <= 0 ||
        cell_count > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        descriptor.blocked_bitmap().size() !=
            static_cast<std::size_t>(cell_count)) {
        error = "blocked bitmap size does not match grid dimensions";
        return false;
    }
    const auto in_bounds = [&](int gx, int gy) {
        return gx >= 0 && gx < static_cast<int>(descriptor.grid_columns()) &&
               gy >= 0 && gy < descriptor.grid_rows();
    };
    if (!in_bounds(descriptor.start_grid_x(), descriptor.start_grid_y()) ||
        !in_bounds(descriptor.goal_grid_x(), descriptor.goal_grid_y())) {
        error = "map start or goal is outside the grid";
        return false;
    }
    for (const unsigned char value : descriptor.blocked_bitmap()) {
        if (value > 1) {
            error = "blocked bitmap contains a non-boolean value";
            return false;
        }
    }
    return true;
}

}  // namespace

bool ValidateMazeMapDescriptor(const maze::MapDescriptor& descriptor,
                               const std::string& expected_map_id,
                               ValidatedMazeMap& validated,
                               std::string& error) {
    validated = ValidatedMazeMap{};
    error.clear();
    if (expected_map_id.empty() || descriptor.map_id() != expected_map_id ||
        !BasicShapeValid(descriptor, error)) {
        if (error.empty()) error = "map identity is invalid";
        return false;
    }

    const int cols = static_cast<int>(descriptor.grid_columns());
    const int rows = descriptor.grid_rows();
    const int cell_count = cols * rows;
    validated.blocked.assign(static_cast<std::size_t>(cell_count), false);
    for (int index = 0; index < cell_count; ++index) {
        validated.blocked[static_cast<std::size_t>(index)] =
            static_cast<unsigned char>(descriptor.blocked_bitmap()[index]) == 1;
    }
    const auto walkable = [&](int gx, int gy) {
        return gx >= 0 && gx < cols && gy >= 0 && gy < rows &&
               !validated.blocked[
                   static_cast<std::size_t>(gy * cols + gx)];
    };
    if (!walkable(descriptor.start_grid_x(), descriptor.start_grid_y()) ||
        !walkable(descriptor.goal_grid_x(), descriptor.goal_grid_y())) {
        error = "map blocks the start or goal cell";
        return false;
    }

    validated.geodesic_distance.assign(
        static_cast<std::size_t>(cell_count), -1);
    const int goal = descriptor.goal_grid_y() * cols +
                     descriptor.goal_grid_x();
    std::deque<int> queue;
    validated.geodesic_distance[static_cast<std::size_t>(goal)] = 0;
    queue.push_back(goal);
    while (!queue.empty()) {
        const int current = queue.front();
        queue.pop_front();
        const int gx = current % cols;
        const int gy = current / cols;
        for (const auto& direction : kDirections) {
            const int dx = direction[0];
            const int dy = direction[1];
            const int nx = gx + dx;
            const int ny = gy + dy;
            if (!walkable(nx, ny)) continue;
            if (dx != 0 && dy != 0 &&
                (!walkable(gx + dx, gy) || !walkable(gx, gy + dy))) {
                continue;
            }
            const int next = ny * cols + nx;
            if (validated.geodesic_distance[
                    static_cast<std::size_t>(next)] >= 0) {
                continue;
            }
            validated.geodesic_distance[static_cast<std::size_t>(next)] =
                validated.geodesic_distance[
                    static_cast<std::size_t>(current)] + 1;
            queue.push_back(next);
        }
    }
    const int start = descriptor.start_grid_y() * cols +
                      descriptor.start_grid_x();
    validated.shortest_action_steps =
        validated.geodesic_distance[static_cast<std::size_t>(start)];
    if (validated.shortest_action_steps <= 0) {
        error = "map is unreachable";
        return false;
    }
    validated.max_finite_distance = 0;
    for (const int distance : validated.geodesic_distance) {
        if (distance >= 0) {
            validated.max_finite_distance =
                std::max(validated.max_finite_distance, distance);
        }
    }
    if (validated.max_finite_distance <= 0) {
        error = "map has no finite geodesic distance range";
        return false;
    }
    error.clear();
    return true;
}
