#!/usr/bin/env python3
"""Verify that the configured Reward V4 identities are reproducible."""

import hashlib
import json
import sys
from pathlib import Path


def canonical_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_flat_yaml(path):
    result = {}
    section = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line[0].isspace() and line.endswith(":"):
            section = line[:-1]
            result[section] = {}
            continue
        if section is None or ":" not in line:
            continue
        key, raw_value = line.strip().split(":", 1)
        value = raw_value.strip().strip('"').strip("'")
        result[section][key] = value
    return result


def number(section, key):
    return float(section[key])


def integer(section, key):
    return int(section[key])


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    require(len(sys.argv) == 2, "server config path is required")
    config = parse_flat_yaml(sys.argv[1])
    semantics_cfg = config["training_semantics"]
    reward_cfg = config["reward"]
    task_cfg = config["task"]

    reward_schema = {
        "schema_id": "maze.reward.v4",
        "schema_version": 1,
        "distance_source": "aiserver.hidden.reverse_bfs.unit_cost.v1",
        "episode_distance_normalizer": (
            "episode_start_geodesic_distance"
        ),
        "learning_reward_sum": [
            "goal_reward",
            "timeout_penalty",
            "geodesic_progress",
            "first_visit_bonus",
            "wasted_action_penalty",
        ],
        "components": {
            "goal_reward": "goal_reached ? goal_reward : 0",
            "timeout_penalty": "time_limit ? timeout_penalty : 0",
            "geodesic_progress": (
                "progress_budget * (d_prev - d_curr) / d0"
            ),
            "first_visit_bonus": (
                "new_cell_after_move ? min(stage_budget / d0, "
                "remaining_stage_budget) : 0"
            ),
            "wasted_action_penalty": (
                "position_unchanged ? wasted_action_penalty : 0"
            ),
        },
        "invalid_transition": (
            "reject if d0 <= 0 or abs(d_prev - d_curr) > 1"
        ),
    }
    reward_digest = canonical_digest(reward_schema)
    require(
        semantics_cfg["reward_schema_id"] == "maze.reward.v4",
        "Reward schema ID is not V4",
    )
    require(
        semantics_cfg["reward_schema_digest"] == reward_digest,
        "Reward schema digest is not canonical",
    )

    semantics = {
        "training_contract_id": semantics_cfg["training_contract_id"],
        "observation_schema": {
            "schema_id": semantics_cfg["observation_schema_id"],
            "schema_version": integer(
                semantics_cfg, "observation_schema_version"
            ),
            "canonical_digest": semantics_cfg[
                "observation_schema_digest"
            ],
        },
        "action_schema": {
            "schema_id": semantics_cfg["action_schema_id"],
            "schema_version": integer(
                semantics_cfg, "action_schema_version"
            ),
            "canonical_digest": semantics_cfg["action_schema_digest"],
        },
        "reward_schema": {
            "schema_id": semantics_cfg["reward_schema_id"],
            "schema_version": integer(
                semantics_cfg, "reward_schema_version"
            ),
            "canonical_digest": reward_digest,
        },
        "policy_distribution_schema_id": semantics_cfg[
            "policy_distribution_schema_id"
        ],
        "model_architecture_id": semantics_cfg["model_architecture_id"],
    }
    require(
        semantics_cfg["semantics_digest"] == canonical_digest(semantics),
        "training semantics digest is not canonical",
    )

    reward = {
        "goal_reward": number(reward_cfg, "goal_reward"),
        "timeout_penalty": number(reward_cfg, "timeout_penalty"),
        "progress_budget": number(reward_cfg, "progress_budget"),
        "stage_8x_first_visit_budget": number(
            reward_cfg, "stage_8x_first_visit_budget"
        ),
        "stage_4x_first_visit_budget": number(
            reward_cfg, "stage_4x_first_visit_budget"
        ),
        "stage_2x_first_visit_budget": number(
            reward_cfg, "stage_2x_first_visit_budget"
        ),
        "wasted_action_penalty": number(
            reward_cfg, "wasted_action_penalty"
        ),
    }
    require(
        reward["timeout_penalty"]
        + reward["progress_budget"]
        + reward["stage_8x_first_visit_budget"]
        < 0.0,
        "maximum 8x failure budget must remain negative",
    )
    require(
        reward["stage_8x_first_visit_budget"]
        >= reward["stage_4x_first_visit_budget"]
        >= reward["stage_2x_first_visit_budget"]
        == 0.0,
        "first-visit curriculum budgets are invalid",
    )

    task = {
        "task_contract_id": task_cfg["task_contract_id"],
        "task_id": task_cfg["task_id"],
        "task_revision": integer(task_cfg, "task_revision"),
        "agent_num": integer(task_cfg, "agent_num"),
        "fixed_map_id": task_cfg["fixed_map_id"],
        "fixed_map_checksum_sha256": task_cfg[
            "fixed_map_checksum_sha256"
        ],
        "action_rule_id": task_cfg["action_rule_id"],
        "shortest_action_steps": integer(
            task_cfg, "shortest_action_steps"
        ),
        "reward_schema_id": semantics_cfg["reward_schema_id"],
        "reward_schema_digest": reward_digest,
        "reward": reward,
    }
    require(task["task_revision"] == 2, "Maze task revision must be 2")
    require(
        task_cfg["task_config_digest"] == canonical_digest(task),
        "Maze task digest is not canonical",
    )


if __name__ == "__main__":
    main()
