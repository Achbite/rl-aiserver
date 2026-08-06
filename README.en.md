# RL AIServer

English | [简体中文](README.md)

C++ environment-interaction, inference, trajectory assembly, and asynchronous sample egress service. Training mode sends samples to LocalSampleService in the Learner Pod, while `local-test` uses the model embedded in the image.

## Quick Start

Build Contracts and explicitly refresh the repository-local protocol snapshot:

```bash
(cd ../rl-contracts && bash build_artifact.sh)
artifact=../.workspace/artifacts/rl-contracts/0.8.0/linux-arm64
cp "${artifact}/common.proto" "${artifact}/training.proto" \
   "${artifact}/maze_task.proto" proto/
cp "${artifact}"/cpp/common.pb.{cc,h} proto/
cp "${artifact}"/cpp/training.pb.{cc,h} proto/
cp "${artifact}"/cpp/training.grpc.pb.{cc,h} proto/
cp "${artifact}"/cpp/maze_task.pb.{cc,h} proto/
cp "${artifact}"/cpp/maze_task.grpc.pb.{cc,h} proto/
cp "${artifact}/manifest.json" proto/
```

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

## License

[MIT License](LICENSE)
