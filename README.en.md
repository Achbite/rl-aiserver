# RL AIServer

[简体中文](README.md) | English

AIServer provides static model evaluation plus training inference, per-Agent
R-PIN segments, GAE/value targets, and processed-transition delivery. For local
training, start it after Learner is ready and connect Client afterwards.

Local training runs only three containers: Learner, AIServer, and Client.
`make shell` is a host command. When `aiserver-dev` is absent it builds the
development image and creates and starts the container. When the container
exists, it starts it only if needed and enters it directly. It never reads the
Contracts repository or synchronizes or replaces this checkout's `proto/`. A
complete three-container workspace has these sibling directories:

```text
workspace/
  rl-contracts/
  rl-sample-pool/
  rl-model-distributor/
  rl-learner/
  rl-aiserver/
  maze-client/
```

The first three repositories add no runtime container. Sample Pool and Model
Distributor are staged only into Learner. `rl-contracts` changes the Client and
AIServer Maze Task Proto only through an explicit protocol-sync command. See
[rl-framework](https://github.com/Achbite/rl-framework) for the startup order.

## 1. Development container, incremental build, and tests

```bash
# Host: build/create when absent; otherwise reuse and enter directly
make shell

# Inside the container: build and test are explicit, separate entrypoints
./build.sh
bash ./test.sh

# The host can also reuse the same container for a build
make build

# Explicitly refresh after Dockerfile.dev, toolchain, environment, or mount changes
make dev-refresh
```

The development image does not inherit an old runtime image and uses a
persistent ccache volume. `ninja: no work to do.` means no source changed; it
does not automatically rerun tests. Tests may be started only from the
repository root with `bash ./test.sh`; `build.sh`, Docker image builds, and
other wrappers do not run them implicitly. Run `make shell` only on the host.
`make dev-image` rebuilds only the image and never replaces an existing
container. `make dev-refresh` rebuilds the image and recreates the container. It
refuses while AIServer, tests, or a build are active. None of these entrypoints
synchronizes protocols.

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

# Deterministic evaluation from an explicit ONNX model file
bash ./run.sh --config configs/server_config.yaml --workload evaluation \
  --evaluation-model /absolute/path/model.onnx

# Training; config supplies defaults and CLI explicitly overrides Learner endpoints
bash ./run.sh --config configs/server_config.yaml --workload training \
  --sample-distributor maze-learner:9100 \
  --model-distributor maze-learner:9200
```

The final evaluation config/CLI value must point to a non-empty, regular,
non-symlink ONNX file. Its filename and parent-directory layout are not part of
the AIServer-Client or model-distribution contract. AIServer does not read a
neighboring manifest or accept a directory entrypoint. `run.sh` only supervises
the process and propagates its exit status; only the C++ config/CLI layer
interprets business arguments.

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

The Training Contract and
`OpenSessionRsp.environment.action_mask_mode` explicitly select `disabled` or
`required`. With masks disabled, Client state and training samples carry none.
When required, AIServer validates the action dimension, masks unavailable logits,
and carries the same mask into the transition consumed by Learner. It is not a
mandatory default capability.

AIServer validates segment continuity, close reason, termination semantics, and
bootstrap exactly once when it closes the segment. These producer-internal
facts are absent from `ProcessedTransition`; Learner and SamplePool receive only
the final PPO observation/action/log-probability/value/advantage/value-target.
A prepared and acknowledged model activates only at each Agent's next segment.

The same runtime image exposes a read-only model diagnostic for tensor-contract
and finite-inference checks:

```bash
/opt/rl/aiserver/bin/maze_aiserver \
  --inspect-model /absolute/path/model.onnx
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

The runtime image compiles the current worktree and repository-local `proto/`.
Normal builds do not read the Contracts repository or compare Client/AIServer
source, generator, hash, or platform identities. Only when intentionally
adopting the current Maze release should you run the Framework command and review
this repository's diff:

```bash
(cd ../rl-framework && bash sync_maze_protocol.sh)
```

Then build the current source with a project tag from the host:

```bash
RL_PROJECT_IMAGE_TAG=maze-tag-001 bash build_image.sh
```

The build never consumes development artifacts or a development-container build
directory. AIServer still validates model/training tensor semantics against the
fields and digest in its repository-owned Training Contract, but it does not use
a central manifest package, platform, or cross-repository hash to lock Client
communication. The full image reference is `rl-training/aiserver:maze-tag-001`,
and a later tuning build may overwrite the same tag.

## 5. Refresh or remove the development container

```bash
make dev-refresh
make dev-clean
```

`dev-refresh` preserves source and the ccache volume while replacing the
development image/container environment. `dev-clean` removes the development
container. Neither command synchronizes protocols.

## 6. Default addresses

| Service | Address |
| --- | --- |
| AIServer gRPC | `0.0.0.0:9002` |
| Learner Sample Pool | `maze-learner:9100` |
| Learner Model Distributor | `maze-learner:9200` |

## License

[MIT License](LICENSE)
