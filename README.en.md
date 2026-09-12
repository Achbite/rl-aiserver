# RL AIServer

Proto and its generated artifacts are the only shared Client–AIServer communication contract.
`TrainingTaskService` implements the common RPC lifecycle; `maze.sdk.pb.h` is generated from the Proto
service descriptor. Shared task components live in `src/task/`, while Maze-specific implementations
live in the sibling `src/maze/` directory. Shared components are organized by function:

| Directory | Responsibility |
| --- | --- |
| `protocol/` | Shared RPC lifecycle and training protocol types |
| `config/` | Shared training configuration types |
| `runtime/` | Training runtime and transactions |
| `session/` | Session management and shared training state |
| `inference/` | ONNX inference |
| `policy/` | Action sampling |
| `reward/` | Shared `RewardResult` interface |
| `sample/` | Transitions, GAE and sample delivery |
| `model/` | Model retrieval, activation and version boundaries |
| `metrics/` | Shared metric registration, window statistics and transport |

`src/maze/` contains `protocol/`, `config/`, `observation/`, `reward/`, `environment/`, `episode/` and
`metrics/`. Maze grids, rays, terminal reasons, reward formulas, metric fields and deployment defaults
belong there. Shared `task/` code does not name Maze types or import Maze Proto.

`main/main.cpp` composes the current task through `maze/task_entry.h`; `src/maze/sources.cmake` lists
its sources and generated Proto sources. Task Proto and generated artifacts remain in
`proto/maze/`. The production build uses RL-SDK's `rl_sdk_generate_task` to compile
the local task Proto and its imports into `build/`, independently from the Client's
compilation of the same contract. It does not overwrite the synchronized snapshot;
Python Protobuf is a build-time generator dependency. The application is
`rl_aiserver`. Maze adapters call shared task components, which
receive task implementations through template parameters.

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
RL_CLIENT_SOURCE_DIR=/workspace/maze-client bash ./test.sh

# The host can also reuse the same container for a build
make build

# Explicitly refresh after Dockerfile.dev, toolchain, environment, or mount changes
make dev-refresh
```

`RL_CLIENT_SOURCE_DIR` points to Client sources mounted read-only in the test container. Only the registered RPC test links the actual environment and action receipts; production builds have no such dependency. Fixed ONNX fixtures verify communication, actions, GAE values and error propagation, not training outcomes.

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
compiled in `src/maze/reward/reward.cpp`; runtime YAML must not contain a `reward:`
tuning section. Model I/O dimensions come from `model.expected_obs_dim` and
`model.expected_action_dim`, rollout values come from `rollout`, and action
sampling plus the optional mask mode come from `policy`. These are owned by the
current AIServer task implementation and are not derived from an external contract
file.

Training uses `rollout.gamma`, `rollout.gae_lambda`, and `rollout.tmax`.
AIServer pins a behavior model independently for each Agent, closes the segment
at the configured TMax, computes unnormalised
GAE/value targets, and submits the resulting items in batches through its
in-process SampleDistributor. The only authority for the actual Agent count is
`environment.agent_count/RL_AISERVER_AGENT_COUNT`; `server.max_agents` is only
a capacity limit. Client, task configuration, Learner, and SamplePool expose no
second Agent-count authority.

`policy.action_mask_mode` selects `disabled` or `required`, and AIServer reports
that choice to Client through `OpenSessionRsp.environment.action_mask_mode`. With
masks disabled, Client state and training samples carry none.
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
/opt/rl/aiserver/bin/rl_aiserver \
  --inspect-model /absolute/path/model.onnx \
  --observation-dim 17 \
  --action-count 9
```

The inspector validates explicit dimensions and emits tensor information. It loads
no service configuration, opens no port, and hard-codes no task dimensions.

Startup discovery waits for the Distributor and the first model within
`model.startup_timeout_ms`. Once a model is selected, a download, ONNX preparation,
or cache publication failure sends a `FAILED` ACK with the original stage and
cause, then ends this startup. It does not repeatedly load the failed candidate.
An unconfirmed failure ACK is also retained in the local error.

During model updates, each candidate uses `model.startup_timeout_ms` as its total
recovery budget starting with the first LOADED ACK; RPC deadlines and retries
respect the remaining budget. Recovery keeps the exact model and Distributor
authority. `model_feedback` distinguishes `ack_pending`, `ack_rejected`, and
`ack_unconfirmed`. When the budget expires, AIServer preserves the unknown ACK
outcome, enters `DEGRADED` with model state `FAILED`, stops model I/O, and
rejects new inference requests. The same maintenance thread still handles Client
Session expiry. SampleDistributor retains ownership of sending and draining
committed samples; model or inference failures do not become sample transport faults.
AIServer never overwrites a possibly applied LOADED
ACK with FAILED, treats an unknown outcome as rejection, or selects a substitute model.

## 3. Training cache

AIServer uses the private `cache` under `model.local_train_dir` and neither
receives nor interprets platform `task_id/run_id`. A `.aiserver.lock` prevents
two AIServers from concurrently using the same directory. Startup does not scan,
recover, or backfill historical cache entries. Only model steps requested from
the Distributor and validated by the current process enter its in-memory index
and pruning scope. If a requested step's destination already exists, AIServer
accepts it only when its lineage/step and declared size match the current model;
all other
existing directories are neither startup facts nor migration/deletion targets:

```bash
bash ./run.sh --config configs/server_config.yaml --workload training
```

Models always come from the isolated training invocation's Learner Model
Distributor. AIServer discovers the current training run's model lineage from
Distributor status and tracks switches by lineage/step during that service
lifecycle. AIServer never starts training from a local savepoint and never removes
its cache on a normal stop.

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
directory. AIServer uses its configured model dimensions, rollout and policy values
plus the Proto fields directly. It reads no Training Contract file and uses no
central manifest, platform, package version, or cross-repository hash to lock Client
or Learner-side components. The full image reference is
`rl-training/aiserver:maze-tag-001`,
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
