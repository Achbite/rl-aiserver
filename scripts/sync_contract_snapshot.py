#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from verify_contract_snapshot import (
    LOCAL_MANIFEST_SCHEMA,
    SNAPSHOT_FILES,
    TASK_PROTOCOL,
    verify_artifact,
    verify_snapshot,
)


def sync_snapshot(artifact_root: Path, target_root: Path, version: str) -> dict:
    manifest = verify_artifact(artifact_root, version)
    target_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".contract-snapshot-", dir=target_root.parent
    ) as temporary:
        stage = Path(temporary)
        for artifact_name, local_name in SNAPSHOT_FILES.items():
            target = stage / local_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact_root / artifact_name, target)
        local_manifest = {
            "schema_version": LOCAL_MANIFEST_SCHEMA,
            "task_protocol": TASK_PROTOCOL,
            "source_release": {
                "package": manifest["package"],
                "version": manifest["version"],
            },
            "files": sorted(SNAPSHOT_FILES.values()),
        }
        (stage / "manifest.json").write_text(
            json.dumps(local_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for local_name in (
            "training.proto",
            "training.pb.cc",
            "training.pb.h",
            "training.grpc.pb.cc",
            "training.grpc.pb.h",
        ):
            shutil.copyfile(target_root / local_name, stage / local_name)
        verify_snapshot(stage)

        for local_name in SNAPSHOT_FILES.values():
            target = target_root / local_name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / local_name, target)
        os.replace(stage / "manifest.json", target_root / "manifest.json")

    verify_snapshot(target_root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly synchronize the Maze task protocol snapshot"
    )
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    manifest = sync_snapshot(
        args.artifact_dir.resolve(),
        args.target_dir.resolve(),
        args.version,
    )
    print(
        "Maze task protocol synchronized: "
        f"release={manifest['version']} protocol=rl.task.maze/1"
    )


if __name__ == "__main__":
    main()
