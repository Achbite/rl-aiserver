# RL AIServer

[简体中文](README.md) | English

AIServer provides training inference, trajectory assembly, and sample delivery. Start full training from [rl-framework](../rl-framework/README.en.md).

## 1. Prepare Contracts and the smoke model

```bash
(cd ../rl-contracts && bash build_artifact.sh)
bash scripts/sync_contract_snapshot.sh
```

The runtime image also needs the smoke model produced by Learner:

```bash
(cd ../rl-learner && RL_LEARNER_IMAGE_TAG=training-001 bash build_image.sh)
```

## 2. Build the runtime image

```bash
RL_AISERVER_IMAGE_TAG=training-001 bash build_image.sh
```

## 3. Incremental build and tests

```bash
# Build the development image from the runtime image
AISERVER_IMAGE_TAG=training-001 make dev-image

# Incrementally build only the main executable; do not run CTest
make build

# Explicitly build test targets and run CTest
make test

# Enter the development container
make shell
```

The development container uses a persistent ccache volume. `ninja: no work to do.` means no source changed; it does not automatically rerun tests.

## 4. Run modes

Inside the development container:

```bash
# Random-model inference-chain check
bash ./run.sh local-test

# Training; normally started by Framework with Learner addresses
bash ./run.sh training

# Local model evaluation
bash ./run.sh model-evaluation
```

Before local evaluation, point `model.evaluation_dir` in `configs/server_config.yaml` to a directory containing `SaveModel.onnx`:

```yaml
model:
  evaluation_dir: "models/evaluation/000200"
```

## 5. Training cache

A normal stop preserves `models/local-train`. Framework passes `--new-run` only for a new Run; that option clears the old AIServer training cache:

```bash
bash ./run.sh training --new-run
```

Training models always come from Learner Model Distributor. AIServer never starts training from a local savepoint and never removes its cache on a normal stop.

## 6. Default addresses

| Service | Address |
| --- | --- |
| AIServer gRPC | `0.0.0.0:9002` |
| Learner Sample Pool | `maze-learner:9100` |
| Learner Model Distributor | `maze-learner:9200` |

## License

[MIT License](LICENSE)
