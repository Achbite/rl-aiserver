# RL AIServer

English | [简体中文](README.md)

C++ environment-interaction, inference, trajectory assembly, and asynchronous sample egress service. Training mode sends samples to LocalSampleService in the Learner Pod, while `local-test` uses the model embedded in the image.

## Quick Start

Build Contracts, then synchronize the repository-local protocol snapshot from
the immutable artifact selected by an explicit version and platform:

```bash
(cd ../rl-contracts && bash build_artifact.sh)
bash scripts/sync_contract_snapshot.sh
```

The synchronization entrypoint reads the explicit `0.10.0` and `linux/arm64`
identity from `artifact_versions.env`, verifies the manifest, every artifact
file, and the staged snapshot before and after replacement. It neither discovers
`latest` nor invokes a host `protoc` to regenerate code.

Build the image:

```bash
RL_AISERVER_IMAGE_TAG=training-001 bash build_image.sh
```

Enter the development container and start the inference smoke test:

```bash
make shell
bash ./run.sh local-test
```

Start training mode:

```bash
bash ./run.sh training
```

For local model evaluation, set `model.evaluation_dir` in
`configs/server_config.yaml` to the savepoint directory. The directory must
contain the fixed filename `SaveModel.onnx`:

```yaml
model:
  evaluation_dir: "models/evaluation/000200"
```

```bash
bash ./run.sh model-evaluation
```

Use `rl-framework` to start the complete workflow.

## Run Modes

```text
1 / training
2 / local-test
3 / model-evaluation
4 / map-validation
```

The default AIServer port is `9002`. Training samples are sent to `maze-learner:9100`, and models are fetched from `maze-learner:9200`.

Training activates a new model only after every active Agent in the current
AIServer reaches a fragment boundary; there is no cross-Server-Pod switch
barrier. Every `SampleBatch` carries the `BehaviorPolicyReference` that produced
it. Evaluation pins the full model identity for the complete Episode and
releases it only after commit or abort.

## License

[MIT License](LICENSE)
