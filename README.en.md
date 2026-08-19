# RL AIServer

[简体中文](README.md) | English

AIServer provides static model evaluation plus training inference, trajectory
assembly, and sample delivery. For the A3 local chain, developers start Learner,
AIServer, and Client separately; Framework no longer orchestrates runtime.

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

Inside the development container:

```bash
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
`--workload` only overrides that field. Reward V4 formulas and numeric values
are compiled in `src/ai/maze_reward.cpp`; runtime YAML must not contain a
`reward:` tuning section and retains only the reward schema identity.

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
two AIServers from concurrently using the same directory. A normal restart
validates and recovers an existing valid cache instead of deleting or rejecting
it merely because it is non-empty:

```bash
bash ./run.sh --config configs/server_config.yaml --workload training
```

Models always come from the isolated training invocation's Learner Model Distributor. AIServer discovers the internal model lineage from Distributor status and pins the first lineage; a different lineage in the same service lifetime fails closed. AIServer never starts training from a local savepoint and never removes its cache on a normal stop.

## 4. Formal artifacts and image

Only after Level 1/2 pass, user review, and clean savepoints may the host sync
the formal Contracts artifact and run `bash build_image.sh`. The formal build
requires clean runtime repositories and never consumes development artifacts or
a development-container build directory.

## 5. Default addresses

| Service | Address |
| --- | --- |
| AIServer gRPC | `0.0.0.0:9002` |
| Learner Sample Pool | `maze-learner:9100` |
| Learner Model Distributor | `maze-learner:9200` |

## License

[MIT License](LICENSE)
