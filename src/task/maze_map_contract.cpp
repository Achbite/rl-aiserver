#include "task/maze_map_contract.h"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>

namespace {

constexpr const char* kActionRuleId =
    "maze.action.9-way.no-corner-cut";
constexpr int kDirections[8][2] = {
    {0, 1}, {1, 1}, {1, 0}, {1, -1},
    {0, -1}, {-1, -1}, {-1, 0}, {-1, 1},
};

bool IsLowerHexSha256(const std::string& value) {
    if (value.size() != 64) return false;
    return std::all_of(value.begin(), value.end(), [](char character) {
        return (character >= '0' && character <= '9') ||
               (character >= 'a' && character <= 'f');
    });
}

std::string Sha256(const std::string& payload) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (!context) return "";
    bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
              EVP_DigestUpdate(
                  context, payload.data(), payload.size()) == 1;
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int size = 0;
    if (ok) ok = EVP_DigestFinal_ex(context, digest.data(), &size) == 1;
    EVP_MD_CTX_free(context);
    if (!ok) return "";
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < size; ++index) {
        output << std::setw(2)
               << static_cast<unsigned int>(digest[index]);
    }
    return output.str();
}

void AppendU32(std::string& payload, std::uint32_t value) {
    payload.push_back(static_cast<char>((value >> 24U) & 0xffU));
    payload.push_back(static_cast<char>((value >> 16U) & 0xffU));
    payload.push_back(static_cast<char>((value >> 8U) & 0xffU));
    payload.push_back(static_cast<char>(value & 0xffU));
}

void AppendI32(std::string& payload, std::int32_t value) {
    AppendU32(payload, static_cast<std::uint32_t>(value));
}

bool BasicShapeValid(const maze::MapDescriptor& descriptor,
                     std::string& error) {
    if (descriptor.grid_columns() == 0 || descriptor.grid_rows() == 0 ||
        descriptor.grid_size_microunits() == 0 ||
        descriptor.action_rule_id() != kActionRuleId) {
        error = "unsupported map format, dimensions, or action rule";
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
    return true;
}

}  // namespace

std::string CanonicalMazeMapChecksum(const maze::MapDescriptor& descriptor,
                                     std::string& error) {
    if (!BasicShapeValid(descriptor, error)) return "";
    std::string payload("rl.task.maze.map\0", 17);
    AppendU32(payload, descriptor.grid_columns());
    AppendU32(payload, descriptor.grid_rows());
    AppendU32(payload, descriptor.grid_size_microunits());
    AppendI32(payload, descriptor.start_grid_x());
    AppendI32(payload, descriptor.start_grid_y());
    AppendI32(payload, descriptor.goal_grid_x());
    AppendI32(payload, descriptor.goal_grid_y());
    AppendU32(payload,
              static_cast<std::uint32_t>(descriptor.blocked_bitmap().size()));
    for (const unsigned char value : descriptor.blocked_bitmap()) {
        if (value > 1) {
            error = "blocked bitmap contains a non-boolean value";
            return "";
        }
        payload.push_back(static_cast<char>(value));
    }
    AppendU32(payload,
              static_cast<std::uint32_t>(descriptor.action_rule_id().size()));
    payload.append(descriptor.action_rule_id());
    const std::string checksum = Sha256(payload);
    if (checksum.empty()) error = "failed to calculate canonical map checksum";
    return checksum;
}

bool ValidateMazeMapDescriptor(const maze::MapDescriptor& descriptor,
                               const std::string& expected_map_id,
                               const std::string& expected_checksum,
                               ValidatedMazeMap& validated,
                               std::string& error) {
    if (expected_map_id.empty() || descriptor.map_id() != expected_map_id ||
        !IsLowerHexSha256(expected_checksum) ||
        descriptor.canonical_digest().algorithm() !=
            common::DIGEST_ALGORITHM_SHA256 ||
        !IsLowerHexSha256(descriptor.canonical_digest().hex())) {
        error = "map identity or expected checksum is invalid";
        return false;
    }
    const std::string computed =
        CanonicalMazeMapChecksum(descriptor, error);
    if (computed.empty() || computed != descriptor.canonical_digest().hex() ||
        computed != expected_checksum) {
        error = "canonical map checksum does not match assigned task";
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
    if (validated.shortest_action_steps <= 0 ||
        descriptor.shortest_action_steps() !=
            validated.shortest_action_steps) {
        error = "map is unreachable or shortest_action_steps is incorrect";
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
    validated.checksum_sha256 = computed;
    return true;
}
