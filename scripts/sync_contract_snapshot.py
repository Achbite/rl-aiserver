#!/usr/bin/env python3

import argparse
import os
import shutil
import tempfile
from pathlib import Path


SNAPSHOT_FILES = {
    "task-maze": {
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
    },
    "training": {
        "common.proto": "common.proto",
        "training.proto": "training.proto",
        "cpp/common.pb.cc": "common.pb.cc",
        "cpp/common.pb.h": "common.pb.h",
        "cpp/training.pb.cc": "training.pb.cc",
        "cpp/training.pb.h": "training.pb.h",
        "cpp/training.grpc.pb.cc": "training.grpc.pb.cc",
        "cpp/training.grpc.pb.h": "training.grpc.pb.h",
    },
}


def require_regular_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"required protocol file is missing: {path}")


def sync_snapshot(artifact_root: Path, target_root: Path, profile: str) -> None:
    files = SNAPSHOT_FILES[profile]
    target_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".protocol-files-", dir=target_root.parent
    ) as temporary:
        stage = Path(temporary)
        for artifact_name, local_name in files.items():
            source = artifact_root / artifact_name
            require_regular_file(source)
            target = stage / local_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

        for local_name in files.values():
            target = target_root / local_name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / local_name, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly synchronize the Maze task protocol files"
    )
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=tuple(SNAPSHOT_FILES),
        required=True,
    )
    args = parser.parse_args()
    sync_snapshot(
        args.artifact_dir.resolve(),
        args.target_dir.resolve(),
        args.profile,
    )
    print(f"{args.profile} protocol files synchronized")


if __name__ == "__main__":
    main()
