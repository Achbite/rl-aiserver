#!/usr/bin/env python3

import hashlib
import json
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: verify_contract_snapshot.py <proto-dir> <version>")
    root = Path(sys.argv[1])
    expected_version = sys.argv[2]
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        fail(f"contract manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"contract manifest is invalid: {exc}")
    if (
        manifest.get("package") != "rl-contracts"
        or manifest.get("version") != expected_version
    ):
        fail(
            "repository-local contract snapshot version mismatch: "
            f"expected {expected_version}, found {manifest.get('version')}"
        )

    files = {
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
    checksums = manifest.get("files", {})
    for artifact_name, local_name in files.items():
        path = root / local_name
        expected = checksums.get(artifact_name)
        if not path.is_file() or not expected:
            fail(f"contract snapshot file is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            fail(f"contract snapshot checksum mismatch: {path}")


if __name__ == "__main__":
    main()
