# RL AIServer

简体中文 | [English](README.en.md)

C++ 环境交互与推理服务。训练模式同时启动 SampleDistributor；推理测试模式使用镜像内置模型。

## 快速开始

先构建 Sample Pool，并装配二进制：

```bash
(cd ../rl-sample-pool && bash build_artifact.sh)
cp -R ../.workspace/artifacts/rl-sample-pool/0.3.0/linux-arm64/. \
  sample-distributor/
```

构建镜像：

```bash
AISERVER_IMAGE_TAG=training-001 bash build_image.sh
```

进入开发容器并启动推理测试：

```bash
make shell
bash ./run.sh inference-smoke
```

启动训练模式：

```bash
bash ./run.sh training
```

完整链路建议从 `rl-framework` 启动。

## 运行模式

```text
1 / training
2 / inference-smoke
3 / model-evaluation
4 / astar-test
```

默认 AIServer 端口为 `9002`，SampleDistributor 端口为 `9100`。

## License

[MIT License](LICENSE)
