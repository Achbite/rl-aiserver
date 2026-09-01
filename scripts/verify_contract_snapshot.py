#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


ARTIFACT_MANIFEST_SCHEMA = "rl.artifact-manifest.v1"
LOCAL_MANIFEST_SCHEMA = "rl.local-task-protocol.v1"
ARTIFACT_PACKAGE = "rl-task-maze-contracts"
TASK_PROTOCOL = {
    "protocol_id": "rl.task.maze",
    "protocol_version": 1,
}
SNAPSHOT_FILES = {
    "common.proto": "common.proto",
    "maze_task.proto": "maze_task.proto",
    "maze_metrics.proto": "maze_metrics.proto",
    "cpp/common.pb.cc": "common.pb.cc",
    "cpp/common.pb.h": "common.pb.h",
    "cpp/maze_task.pb.cc": "maze_task.pb.cc",
    "cpp/maze_task.pb.h": "maze_task.pb.h",
    "cpp/maze_task.grpc.pb.cc": "maze_task.grpc.pb.cc",
    "cpp/maze_task.grpc.pb.h": "maze_task.grpc.pb.h",
    "cpp/maze_metrics.pb.cc": "maze_metrics.pb.cc",
    "cpp/maze_metrics.pb.h": "maze_metrics.pb.h",
    "schemas/maze.episode.metrics.json": "schemas/maze.episode.metrics.json",
    "schemas/maze.episode.metrics.sha256": "schemas/maze.episode.metrics.sha256",
    "schemas/training-contract.json": "schemas/training-contract.json",
    "schemas/training-contract.sha256": "schemas/training-contract.sha256",
}
LOCAL_TRAINING_FILES = (
    "training.proto",
    "training.pb.cc",
    "training.pb.h",
    "training.grpc.pb.cc",
    "training.grpc.pb.h",
)


def fail(message: str) -> None:
    raise SystemExit(message)


def load_manifest(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        fail(f"contract manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"contract manifest is invalid: {exc}")
    if not isinstance(manifest, dict):
        fail(f"contract manifest must be an object: {manifest_path}")
    return manifest


def verify_artifact(root: Path, expected_version: str) -> dict:
    manifest = load_manifest(root)
    if (
        manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA
        or manifest.get("package") != ARTIFACT_PACKAGE
        or manifest.get("version") != expected_version
        or manifest.get("task_protocol") != TASK_PROTOCOL
    ):
        fail(
            "Maze task artifact identity mismatch: "
            f"expected {ARTIFACT_PACKAGE} {expected_version} {TASK_PROTOCOL}, "
            f"found {manifest.get('package')} {manifest.get('version')} "
            f"{manifest.get('task_protocol')}"
        )
    declared = manifest.get("files")
    if not isinstance(declared, list):
        fail("contract manifest files list is invalid")
    for artifact_name in SNAPSHOT_FILES:
        path = root / artifact_name
        if artifact_name not in declared or path.is_symlink() or not path.is_file():
            fail(f"Maze task artifact file is missing or invalid: {path}")
    return manifest


def verify_snapshot(root: Path) -> dict:
    manifest = load_manifest(root)
    if (
        manifest.get("schema_version") != LOCAL_MANIFEST_SCHEMA
        or manifest.get("task_protocol") != TASK_PROTOCOL
    ):
        fail(f"repository-local Maze task protocol manifest is invalid: {root}")
    declared = manifest.get("files")
    if not isinstance(declared, list):
        fail("repository-local Maze task protocol files list is invalid")
    for local_name in SNAPSHOT_FILES.values():
        path = root / local_name
        if local_name not in declared or path.is_symlink() or not path.is_file():
            fail(f"repository-local Maze task protocol file is missing: {path}")
    for local_name in LOCAL_TRAINING_FILES:
        path = root / local_name
        if path.is_symlink() or not path.is_file():
            fail(f"repository-local training protocol file is missing: {path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify AIServer repository-local protocol inputs"
    )
    parser.add_argument("proto_dir", type=Path)
    args = parser.parse_args()
    verify_snapshot(args.proto_dir.resolve())


if __name__ == "__main__":
    main()
