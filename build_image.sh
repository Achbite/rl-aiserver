#!/usr/bin/env bash

set -euo pipefail

image_tag="${RL_PROJECT_IMAGE_TAG:-maze-tag-001}"
AISERVER_IMAGE_NAME="rl-training/aiserver"

if [[ ! "${image_tag}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
    echo "RL_PROJECT_IMAGE_TAG is not a valid Docker tag" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
context_root="${workspace_root}/.workspace/build-contexts/rl-aiserver-$$"
source "${repo_dir}/artifact_versions.env"
contract_dir="${repo_dir}/proto"

bash "${repo_dir}/scripts/verify_source_inventory.sh"

if [ ! -f "${contract_dir}/manifest.json" ] ||
   [ ! -f "${contract_dir}/common.pb.cc" ] ||
   [ ! -f "${contract_dir}/training.pb.cc" ] ||
   [ ! -f "${contract_dir}/training.grpc.pb.cc" ] ||
   [ ! -f "${contract_dir}/maze_task.pb.cc" ] ||
   [ ! -f "${contract_dir}/maze_task.grpc.pb.cc" ]; then
    echo "Repository-local contract snapshot is incomplete: ${contract_dir}" >&2
    exit 1
fi
python3 - \
    "${contract_dir}/manifest.json" \
    "${RL_CONTRACTS_VERSION}" \
    "${RL_CONTRACTS_PLATFORM}" \
    <<'PY'
import hashlib
import json
from pathlib import Path
import sys

contract_path = Path(sys.argv[1])
contract = json.loads(contract_path.read_text(encoding="utf-8"))

def verify_contract_files(root, manifest):
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
    for artifact_name, local_name in files.items():
        path = root / local_name
        expected_checksum = manifest.get("files", {}).get(artifact_name)
        if not path.is_file() or not expected_checksum:
            raise SystemExit(f"Repository-local contract file is missing: {path}")
        actual_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_checksum != expected_checksum:
            raise SystemExit(f"Repository-local contract checksum mismatch: {path}")

if (
    contract.get("schema_version") != 2
    or contract.get("package") != "rl-contracts"
    or contract.get("version") != sys.argv[2]
    or contract.get("platform") != sys.argv[3]
    or contract.get("source_tree_state") != "clean"
):
    raise SystemExit("Contract artifact identity is invalid")
verify_contract_files(contract_path.parent, contract)
PY

stack_identity_tool="${workspace_root}/rl-framework/tools/compute_stack_source_id.py"
if [ ! -f "${stack_identity_tool}" ]; then
    echo "stack source identity tool is missing: ${stack_identity_tool}" >&2
    exit 1
fi
stack_identity_arguments=(
    --workspace-root "${workspace_root}"
)
stack_identity_json="$(
    python3 "${stack_identity_tool}" "${stack_identity_arguments[@]}"
)"
identity_fields="$(
    python3 -c '
import json
import sys

document = json.loads(sys.argv[1])
print("\t".join((
    document["stack_source_id"],
    document["repositories"]["rl-aiserver"],
    document["artifacts"]["rl-contracts"]["artifact_digest"],
    document["artifacts"]["rl-contracts"]["manifest_sha256"],
    document["configs"]["aiserver"]["sha256"],
)))
' "${stack_identity_json}"
)"
IFS=$'\t' read -r \
    stack_source_id component_commit contracts_artifact_digest \
    contracts_manifest_digest component_config_digest \
    <<< "${identity_fields}"
component_contract_tool="${workspace_root}/rl-framework/tools/generate_component_contract.py"
if [ ! -f "${component_contract_tool}" ]; then
    echo "component contract generator is missing: ${component_contract_tool}" >&2
    exit 1
fi
contract_temp_dir="$(mktemp -d)"
trap 'rm -rf "${context_root}" "${contract_temp_dir}"' EXIT
contract_manifest_digest="$(
    python3 "${component_contract_tool}" \
        --component aiserver \
        --output "${contract_temp_dir}/manifest.json" \
        --schema-source "${repo_dir}/component-contract/config.schema.json" \
        --schema-image-path /opt/rl/component-contract/config.schema.json \
        --config-file "main=/opt/rl/aiserver/configs/server_config.yaml=aiserver.yaml=${repo_dir}/configs/server_config.yaml" \
        --contracts-version "${RL_CONTRACTS_VERSION}" \
        --contracts-artifact-digest "${contracts_artifact_digest}" \
        --supported-maps "${repo_dir}/component-contract/supported_maps.json"
)"
image_ref="${AISERVER_IMAGE_NAME}:${image_tag}"

python3 - \
    "${repo_dir}" \
    "${context_root}" <<'PY'
import pathlib
import shutil
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])

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
PY

mkdir -p "${context_root}/_deps/identity"
printf '%s\n' "${stack_identity_json}" \
    > "${context_root}/_deps/identity/stack-source.json"
cp "${contract_temp_dir}/manifest.json" \
    "${context_root}/component-contract/manifest.json"

docker build \
    --label "org.opencontainers.image.revision=${component_commit}" \
    --label "org.rl-training.component=aiserver" \
    --label "org.rl-training.component-commit=${component_commit}" \
    --label "org.rl-training.stack-source-id=${stack_source_id}" \
    --label "org.rl-training.contracts-version=${RL_CONTRACTS_VERSION}" \
    --label "org.rl-training.contracts-artifact-digest=${contracts_artifact_digest}" \
    --label "org.rl-training.contracts-manifest-digest=${contracts_manifest_digest}" \
    --label "org.rl-training.component-config-digest=${component_config_digest}" \
    --label "org.rl-training.component-contract.path=/opt/rl/component-contract/manifest.json" \
    --label "org.rl-training.component-contract.sha256=${contract_manifest_digest}" \
    --label "org.rl-training.project-image-tag=${image_tag}" \
    --tag "${image_ref}" \
    "${context_root}"

printf '%s\n' "${image_ref}"
