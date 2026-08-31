# RL AIServer

[简体中文](README.md) | English

AIServer provides static model evaluation plus training inference, per-Agent
R-PIN segments, GAE/value targets, and processed-transition delivery. For local
training, start it after Learner is ready and connect Client afterwards.

Local training runs only three containers: Learner, AIServer, and Client.
`make shell` is a host command that prepares development artifacts from sibling
source repositories; it does not download those repositories. A fresh workspace
therefore needs at least these sibling directories:

```text
workspace/
  rl-contracts/
  rl-sample-pool/
  rl-model-distributor/
  rl-learner/
  rl-aiserver/
  maze-client/
```

The first three repositories supply development artifacts only and do not add
runtime containers. See [rl-framework](https://github.com/Achbite/rl-framework)
for the complete three-container startup order.

## 1. Development container, incremental build, and tests

```bash
# Host: build or reuse the independent development image and enter it
make shell

# Inside the container: build and test are explicit, separate entrypoints
./build.sh
bash ./test.sh

# The host can also reuse the same container for a build
make build
```

The development image does not inherit an old runtime image and uses a
persistent ccache volume. `ninja: no work to do.` means no source changed; it
does not automatically rerun tests. Tests may be started only from the
repository root with `bash ./test.sh`; `build.sh`, Docker image builds, and
other wrappers do not run them implicitly. Run `make shell` only on the host.

## 2. Run modes

After Learner is running, open a second host terminal:

```bash
# Host
cd /path/to/workspace/rl-aiserver
make shell

# Run the following commands inside the AIServer container
./build.sh

# Show the executable CLI-to-config mapping without starting the service
bash ./run.sh --help

# Deterministic evaluation from an explicit SaveModel.onnx
bash ./run.sh --config configs/server_config.yaml --workload evaluation \
  --evaluation-model /absolute/path/SaveModel.onnx

# Training; config supplies defaults and CLI explicitly overrides Learner endpoints
bash ./run.sh --config configs/server_config.yaml --workload training \
  --sample-distributor maze-learner:9100 \
  --model-distributor maze-learner:9200
```

The final evaluation config/CLI value must point to a regular, non-symlink
`SaveModel.onnx`. AIServer does not read a neighboring manifest or accept a
directory entrypoint. `run.sh` only supervises the process and propagates its
exit status; only the C++ config/CLI layer interprets business arguments.

The default workload is `server.run_mode` in `configs/server_config.yaml`, and
`--workload` only overrides that field. Reward formulas and numeric values are
compiled in `src/ai/maze_reward.cpp`; runtime YAML must not contain a `reward:`
tuning section. The training contract and its observation/action/reward schema
identities come only from the artifact selected by
`contract.training_contract_path`.

Training uses the `RolloutEstimatorProfile` embedded in the model manifest.
AIServer pins a behavior model independently for each Agent, closes the segment
after at most 128 completed transitions by default, computes unnormalised
GAE/value targets, and submits the resulting items in batches through its
in-process SampleDistributor. The only authority for the actual Agent count is
`environment.agent_count/RL_AISERVER_AGENT_COUNT`; `server.max_agents` is only
a capacity limit. Client, task configuration, Learner, and SamplePool expose no
second Agent-count authority.

AIServer validates segment continuity, close reason, termination semantics, and
bootstrap exactly once when it closes the segment. These producer-internal
facts are absent from `ProcessedTransition`; Learner and SamplePool receive only
the final PPO observation/action/log-probability/value/advantage/value-target.
A prepared and acknowledged model activates only at each Agent's next segment.

The same runtime image exposes a read-only model diagnostic for tensor-contract
and finite-inference checks:

```bash
/opt/rl/aiserver/bin/maze_aiserver \
  --inspect-model /absolute/path/SaveModel.onnx
```

The inspector emits only tensor identity; it loads no service configuration and
opens no port.

## 3. Training cache

AIServer uses the private `cache` under `model.local_train_dir` and neither
receives nor interprets platform `task_id/run_id`. A `.aiserver.lock` prevents
two AIServers from concurrently using the same directory. Startup does not scan,
recover, or backfill historical cache entries. Only model steps requested from
the Distributor and validated by the current process enter its in-memory index
and pruning scope. If a requested step's destination already exists, AIServer
accepts it only when it exactly matches the current protobuf manifest; all other
existing directories are neither startup facts nor migration/deletion targets:

```bash
bash ./run.sh --config configs/server_config.yaml --workload training
```

Models always come from the isolated training invocation's Learner Model Distributor. AIServer discovers the internal model lineage from Distributor status and pins the first lineage; a different lineage in the same service lifetime fails closed. AIServer never starts training from a local savepoint and never removes its cache on a normal stop.

## 4. Build the runtime image

The runtime image is built from the current worktree so development can follow
edit, build, validate locally, and only then commit. Git clean/dirty state is
diagnostic provenance, not a build admission gate. The Contracts artifact that is
actually packaged must still be synchronized. Run from the host:

```bash
bash scripts/sync_contract_snapshot.sh
RL_PROJECT_IMAGE_TAG=maze-tag-001 bash build_image.sh
```

The build never consumes development artifacts or a development-container build
directory. The full image reference is `rl-training/aiserver:maze-tag-001`, and a
later tuning build may overwrite the same tag. Existing source-identity labels are
diagnostic artifact facts and no longer determine the image tag.

## 5. Default addresses

| Service | Address |
| --- | --- |
| AIServer gRPC | `0.0.0.0:9002` |
| Learner Sample Pool | `maze-learner:9100` |
| Learner Model Distributor | `maze-learner:9200` |

## License

[MIT License](LICENSE)
