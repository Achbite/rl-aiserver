# RL AIServer

English | [简体中文](README.md)

C++ environment-interaction and inference service. Training mode also starts SampleDistributor, while inference smoke mode uses the model embedded in the image.

## Quick Start

Build Sample Pool and stage the binary:

```bash
(cd ../rl-sample-pool && bash build_artifact.sh)
cp -R ../.workspace/artifacts/rl-sample-pool/0.5.0/linux-arm64/. \
  sample-distributor/
```

Build the image:

```bash
AISERVER_IMAGE_TAG=training-001 bash build_image.sh
```

Enter the development container and start the inference smoke test:

```bash
make shell
bash ./run.sh inference-smoke
```

Start training mode:

```bash
bash ./run.sh training
```

Use `rl-framework` to start the complete workflow.

## Run Modes

```text
1 / training
2 / inference-smoke
3 / model-evaluation
4 / astar-test
```

The default AIServer port is `9002`, and SampleDistributor uses `9100`.

## License

[MIT License](LICENSE)
