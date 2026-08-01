#!/usr/bin/env bash

set -euo pipefail

AISERVER_IMAGE_TAG="${AISERVER_IMAGE_TAG:-training-001}"
AISERVER_IMAGE_NAME="rl-training/aiserver"

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
artifact_root="${workspace_root}/.workspace/artifacts"
context_root="${workspace_root}/.workspace/build-contexts/rl-aiserver-$$"
source "${repo_dir}/artifact_versions.env"
platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
platform_dir="${platform//\//-}"
contract_dir="${repo_dir}/proto"
smoke_model_dir="${artifact_root}/rl-smoke-model/${RL_SMOKE_MODEL_VERSION}/any"

if [ ! -f "${contract_dir}/manifest.json" ] ||
   [ ! -f "${contract_dir}/maze.pb.cc" ] ||
   [ ! -f "${contract_dir}/maze.grpc.pb.cc" ]; then
    echo "Repository-local contract snapshot is incomplete: ${contract_dir}" >&2
    exit 1
fi
if [ ! -f "${smoke_model_dir}/manifest.json" ]; then
    echo "Smoke model artifact is missing. Run ../rl-learner/build_image.sh" >&2
    exit 1
fi

python3 - \
    "${contract_dir}/manifest.json" \
    "${smoke_model_dir}/manifest.json" \
    "${RL_CONTRACTS_VERSION}" \
    "${RL_SMOKE_MODEL_VERSION}" \
    <<'PY'
import hashlib
import json
from pathlib import Path
import sys

contract_path = Path(sys.argv[1])
smoke_path = Path(sys.argv[2])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
smoke = json.loads(smoke_path.read_text(encoding="utf-8"))

def verify_files(root, manifest):
    for relative, expected_checksum in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"Artifact file is missing: {path}")
        actual_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_checksum != expected_checksum:
            raise SystemExit(f"Artifact checksum mismatch: {path}")

def verify_contract_files(root, manifest):
    files = {
        "maze.proto": "maze.proto",
        "cpp/maze.pb.cc": "maze.pb.cc",
        "cpp/maze.pb.h": "maze.pb.h",
        "cpp/maze.grpc.pb.cc": "maze.grpc.pb.cc",
        "cpp/maze.grpc.pb.h": "maze.grpc.pb.h",
    }
    for artifact_name, local_name in files.items():
        path = root / local_name
        expected_checksum = manifest.get("files", {}).get(artifact_name)
        if not path.is_file() or not expected_checksum:
            raise SystemExit(f"Repository-local contract file is missing: {path}")
        actual_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_checksum != expected_checksum:
            raise SystemExit(f"Repository-local contract checksum mismatch: {path}")

if contract.get("package") != "rl-contracts" or contract.get("version") != sys.argv[3]:
    raise SystemExit("Contract artifact identity is invalid")
verify_contract_files(contract_path.parent, contract)
if smoke.get("package") != "rl-smoke-model" or smoke.get("version") != sys.argv[4]:
    raise SystemExit("Smoke model artifact identity is invalid")
if smoke.get("contract_version") != contract["version"]:
    raise SystemExit("Smoke model artifact uses a different contract version")
if not smoke.get("ready"):
    raise SystemExit("Smoke model manifest identity is invalid")
model_path = smoke_path.parent / smoke.get("model_file", "")
if not model_path.is_file():
    raise SystemExit("Smoke model file is missing")
payload = model_path.read_bytes()
if len(payload) != smoke.get("size_bytes"):
    raise SystemExit("Smoke model size does not match its manifest")
if hashlib.sha256(payload).hexdigest() != smoke.get("sha256"):
    raise SystemExit("Smoke model checksum does not match its manifest")
PY

python3 - \
    "${repo_dir}" \
    "${context_root}" \
    "${smoke_model_dir}" <<'PY'
import pathlib
import shutil
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
smoke_model = pathlib.Path(sys.argv[3])

def ignore_runtime_outputs(directory, names):
    ignored = {
        ".git",
        "build",
        "_deps",
        ".DS_Store",
        "models",
        "sample-distributor",
    } & set(names)
    if pathlib.Path(directory) == source:
        ignored.update({"log", "logs"} & set(names))
    return ignored

if target.exists():
    shutil.rmtree(target)
shutil.copytree(
    source,
    target,
    ignore=ignore_runtime_outputs,
)
if not (target / "src/log/logger.h").is_file():
    raise SystemExit("Build context is missing src/log/logger.h")
shutil.copytree(smoke_model, target / "_deps/smoke-model")
PY

trap 'rm -rf "${context_root}"' EXIT
docker build \
    --tag "${AISERVER_IMAGE_NAME}:${AISERVER_IMAGE_TAG}" \
    "${context_root}"

printf '%s\n' "${AISERVER_IMAGE_NAME}:${AISERVER_IMAGE_TAG}"
