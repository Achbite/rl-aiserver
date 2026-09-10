#!/usr/bin/env python3

import argparse
import os
import shutil
import tempfile
from pathlib import Path


SNAPSHOT_FILES = {'task-maze': {'proto/common/identity.proto': 'common/identity.proto',
               'cpp/proto/common/identity.pb.cc': 'common/identity.pb.cc',
               'cpp/proto/common/identity.pb.h': 'common/identity.pb.h',
               'proto/metrics/registry.proto': 'metrics/registry.proto',
               'cpp/proto/metrics/registry.pb.cc': 'metrics/registry.pb.cc',
               'cpp/proto/metrics/registry.pb.h': 'metrics/registry.pb.h',
               'proto/communication/session.proto': 'communication/session.proto',
               'cpp/proto/communication/session.pb.cc': 'communication/session.pb.cc',
               'cpp/proto/communication/session.pb.h': 'communication/session.pb.h',
               'proto/maze/maze.proto': 'maze/maze.proto',
               'cpp/proto/maze/maze.pb.cc': 'maze/maze.pb.cc',
               'cpp/proto/maze/maze.pb.h': 'maze/maze.pb.h',
               'proto/maze/metrics.proto': 'maze/metrics.proto',
               'cpp/proto/maze/metrics.pb.cc': 'maze/metrics.pb.cc',
               'cpp/proto/maze/metrics.pb.h': 'maze/metrics.pb.h',
               'cpp/proto/maze/maze.grpc.pb.cc': 'maze/maze.grpc.pb.cc',
               'cpp/proto/maze/maze.grpc.pb.h': 'maze/maze.grpc.pb.h',
               'cpp/proto/maze/maze.sdk.pb.h': 'maze/maze.sdk.pb.h'},
 'training': {'proto/common/identity.proto': 'common/identity.proto',
              'cpp/proto/common/identity.pb.cc': 'common/identity.pb.cc',
              'cpp/proto/common/identity.pb.h': 'common/identity.pb.h',
              'proto/training/model_identity.proto': 'training/model_identity.proto',
              'cpp/proto/training/model_identity.pb.cc': 'training/model_identity.pb.cc',
              'cpp/proto/training/model_identity.pb.h': 'training/model_identity.pb.h',
              'proto/training/training.proto': 'training/training.proto',
              'cpp/proto/training/training.pb.cc': 'training/training.pb.cc',
              'cpp/proto/training/training.pb.h': 'training/training.pb.h',
              'proto/metrics/registry.proto': 'metrics/registry.proto',
              'cpp/proto/metrics/registry.pb.cc': 'metrics/registry.pb.cc',
              'cpp/proto/metrics/registry.pb.h': 'metrics/registry.pb.h',
              'proto/metrics/catalog.proto': 'metrics/catalog.proto',
              'cpp/proto/metrics/catalog.pb.cc': 'metrics/catalog.pb.cc',
              'cpp/proto/metrics/catalog.pb.h': 'metrics/catalog.pb.h',
              'proto/metrics/transport.proto': 'metrics/transport.proto',
              'cpp/proto/metrics/transport.pb.cc': 'metrics/transport.pb.cc',
              'cpp/proto/metrics/transport.pb.h': 'metrics/transport.pb.h',
              'cpp/proto/training/training.grpc.pb.cc': 'training/training.grpc.pb.cc',
              'cpp/proto/training/training.grpc.pb.h': 'training/training.grpc.pb.h',
              'cpp/proto/metrics/catalog.grpc.pb.cc': 'metrics/catalog.grpc.pb.cc',
              'cpp/proto/metrics/catalog.grpc.pb.h': 'metrics/catalog.grpc.pb.h',
              'cpp/proto/metrics/transport.grpc.pb.cc': 'metrics/transport.grpc.pb.cc',
              'cpp/proto/metrics/transport.grpc.pb.h': 'metrics/transport.grpc.pb.h'}}


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


def sync_sdk(artifact_root: Path, target_root: Path) -> None:
    source = artifact_root / "sdk"
    require_regular_file(source / "CMakeLists.txt")
    require_regular_file(source / "include/rl_sdk/task_client.h")
    target = target_root / "rl_sdk"
    shutil.copytree(source, target, dirs_exist_ok=True)
    # The SDK now has one independent CMake target and include tree.
    for name in ("session.h", "transport.h", "server_command.h", "replay_window.h", "metric_catalog.h"):
        (target / name).unlink(missing_ok=True)


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
    sync_sdk(args.artifact_dir.resolve(), args.target_dir.resolve())
    print(f"{args.profile} protocol files and SDK synchronized")


if __name__ == "__main__":
    main()
