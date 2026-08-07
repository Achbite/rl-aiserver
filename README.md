# RL AIServer

简体中文 | [English](README.en.md)

C++ 环境交互、推理、轨迹组装与异步样本发送服务。训练模式将样本发送至 Learner Pod 的 LocalSampleService；`local-test` 使用镜像内置模型。

## 快速开始

先构建 Contracts，并从固定版本和平台的不可变制品同步仓库内协议快照：

```bash
(cd ../rl-contracts && bash build_artifact.sh)
bash scripts/sync_contract_snapshot.sh
```

同步入口从 `artifact_versions.env` 读取显式的 `0.9.1` 与 `linux/arm64`，在替换前后
校验 manifest、全部制品文件和仓库快照。它不会发现 `latest`，也不会调用本机
`protoc` 重新生成代码。

构建镜像：

```bash
RL_AISERVER_IMAGE_TAG=training-001 bash build_image.sh
```

进入开发容器并启动推理测试：

```bash
make shell
bash ./run.sh local-test
```

启动训练模式：

```bash
bash ./run.sh training
```

评测本地模型时，在 `configs/server_config.yaml` 中将 `model.evaluation_dir`
指向保存点目录。该目录必须包含固定文件名 `SaveModel.onnx`：

```yaml
model:
  evaluation_dir: "models/evaluation/000200"
```

```bash
bash ./run.sh model-evaluation
```

完整链路建议从 `rl-framework` 启动。

## 运行模式

```text
1 / training
2 / local-test
3 / model-evaluation
4 / map-validation
```

默认 AIServer 端口为 `9002`。训练样本默认发送至 `maze-learner:9100`，模型默认从 `maze-learner:9200` 拉取。

训练模式只在当前 AIServer 内所有活跃 Agent 到达 fragment 边界后切换模型，
不要求多个 Server Pod 同步切换。每个 `SampleBatch` 携带实际
`BehaviorPolicyReference`；评测模式在整个 Episode 内固定完整模型身份，
评测结束或中止后才允许切换。

## License

[MIT License](LICENSE)
