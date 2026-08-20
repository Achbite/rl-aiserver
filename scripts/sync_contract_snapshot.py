#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from verify_contract_snapshot import SNAPSHOT_FILES, fail, verify_snapshot


EXPECTED_PACKAGES = [
    "rl.common.v1",
    "rl.training.v1",
    "rl.task.maze.v1",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_artifact(root: Path, version: str, platform: str) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        fail(f"contract artifact manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"contract artifact manifest is invalid: {exc}")

    if (
        manifest.get("schema_version") != 2
        or manifest.get("package") != "rl-contracts"
        or manifest.get("version") != version
        or manifest.get("platform") != platform
        or manifest.get("contract_packages") != EXPECTED_PACKAGES
    ):
        fail(
            "contract artifact identity mismatch: "
            f"expected rl-contracts {version} {platform}"
        )

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        fail("contract artifact files table is invalid")
    for relative, expected in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            fail(f"contract artifact path is unsafe: {relative}")
        path = root / relative_path
        if not path.is_file() or sha256(path) != expected:
            fail(f"contract artifact checksum mismatch: {path}")

    canonical_files = json.dumps(
        files, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    expected_artifact_digest = hashlib.sha256(canonical_files).hexdigest()
    if manifest.get("artifact_digest") != {
        "algorithm": "sha256",
        "hex": expected_artifact_digest,
    }:
        fail("contract artifact digest is invalid")

    generator_path = root / "generator-identity.json"
    try:
        generator = json.loads(generator_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"contract generator identity is invalid: {exc}")
    generator_digest = hashlib.sha256(
        json.dumps(generator, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()
    if manifest.get("generator_identity") != generator_digest:
        fail("contract generator identity checksum mismatch")
    return manifest


def sync_snapshot(
    artifact_root: Path, target_root: Path, version: str, platform: str
) -> dict:
    manifest = verify_artifact(artifact_root, version, platform)
    target_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".contract-snapshot-", dir=target_root.parent
    ) as temporary:
        stage = Path(temporary)
        for artifact_name, local_name in SNAPSHOT_FILES.items():
            (stage / local_name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact_root / artifact_name, stage / local_name)
        shutil.copyfile(artifact_root / "manifest.json", stage / "manifest.json")
        verify_snapshot(stage, version, platform)

        for local_name in SNAPSHOT_FILES.values():
            (target_root / local_name).parent.mkdir(
                parents=True, exist_ok=True
            )
            os.replace(stage / local_name, target_root / local_name)
        os.replace(stage / "manifest.json", target_root / "manifest.json")

    verify_snapshot(target_root, version, platform)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize an exact rl-contracts artifact snapshot"
    )
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--platform", required=True)
    args = parser.parse_args()
    manifest = sync_snapshot(
        args.artifact_dir.resolve(),
        args.target_dir.resolve(),
        args.version,
        args.platform,
    )
    print(
        "contract snapshot synchronized: "
        f"version={manifest['version']} "
        f"platform={manifest['platform']} "
        f"artifact={manifest['artifact_digest']['hex']}"
    )


if __name__ == "__main__":
    main()
