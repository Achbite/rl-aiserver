#!/usr/bin/env python3

import hashlib
import json
import sys
from pathlib import Path


SNAPSHOT_FILES = {
    "common.proto": "common.proto",
    "training.proto": "training.proto",
    "maze_task.proto": "maze_task.proto",
    "cpp/common.pb.cc": "common.pb.cc",
    "cpp/common.pb.h": "common.pb.h",
    "cpp/training.pb.cc": "training.pb.cc",
    "cpp/training.pb.h": "training.pb.h",
    "cpp/training.grpc.pb.cc": "training.grpc.pb.cc",
    "cpp/training.grpc.pb.h": "training.grpc.pb.h",
    "cpp/maze_task.pb.cc": "maze_task.pb.cc",
    "cpp/maze_task.pb.h": "maze_task.pb.h",
    "cpp/maze_task.grpc.pb.cc": "maze_task.grpc.pb.cc",
    "cpp/maze_task.grpc.pb.h": "maze_task.grpc.pb.h",
}


def fail(message: str) -> None:
    raise SystemExit(message)


def verify_snapshot(
    root: Path, expected_version: str, expected_platform: str
) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        fail(f"contract manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"contract manifest is invalid: {exc}")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("package") != "rl-contracts"
        or manifest.get("version") != expected_version
        or manifest.get("platform") != expected_platform
    ):
        fail(
            "repository-local contract snapshot identity mismatch: "
            f"expected rl-contracts {expected_version} {expected_platform}, "
            f"found {manifest.get('package')} {manifest.get('version')} "
            f"{manifest.get('platform')}"
        )

    checksums = manifest.get("files", {})
    if not isinstance(checksums, dict):
        fail("contract manifest files table is invalid")
    for artifact_name, local_name in SNAPSHOT_FILES.items():
        path = root / local_name
        expected = checksums.get(artifact_name)
        if not path.is_file() or not expected:
            fail(f"contract snapshot file is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            fail(f"contract snapshot checksum mismatch: {path}")
    return manifest


def main() -> None:
    if len(sys.argv) != 4:
        fail(
            "usage: verify_contract_snapshot.py "
            "<proto-dir> <version> <platform>"
        )
    verify_snapshot(Path(sys.argv[1]), sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
