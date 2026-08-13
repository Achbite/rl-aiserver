# RL AIServer

简体中文 | [English](README.en.md)

AIServer 提供训练推理、轨迹组装和样本发送。完整训练请从 [rl-framework](../rl-framework/README.md) 启动。

## 1. 准备 Contracts 和 smoke model

```bash
(cd ../rl-contracts && bash build_artifact.sh)
bash scripts/sync_contract_snapshot.sh
```

运行镜像还需要 Learner 生成的 smoke model：

```bash
(cd ../rl-learner && RL_LEARNER_IMAGE_TAG=training-001 bash build_image.sh)
```

## 2. 构建运行镜像

```bash
RL_AISERVER_IMAGE_TAG=training-001 bash build_image.sh
```

## 3. 增量构建与测试

```bash
# 构建开发镜像；指定已经构建的运行镜像
AISERVER_IMAGE_TAG=training-001 make dev-image

# 只增量编译主程序，不运行 CTest
make build

# 显式构建测试目标并运行 CTest
make test

# 进入开发容器
make shell
```

开发容器使用持久 ccache。`ninja: no work to do.` 表示源码未变化，不会自动重复运行测试。

## 4. 运行模式

进入开发容器后：

```bash
# 随机模型推理链路验证
bash ./run.sh local-test

# 训练模式；通常由 Framework 启动并注入 Learner 地址
bash ./run.sh training

# 本地模型评测
bash ./run.sh model-evaluation
```

本地评测前，将 `configs/server_config.yaml` 的 `model.evaluation_dir` 指向包含 `SaveModel.onnx` 的目录：

```yaml
model:
  evaluation_dir: "models/evaluation/000200"
```

## 5. Training 缓存

正常停止保留 `models/local-train` 缓存。开始新 Run 时由 Framework 传入 `--new-run`；仅该参数会清理 AIServer 的旧训练缓存：

```bash
bash ./run.sh training --new-run
```

训练模型始终从 Learner 的 Model Distributor 拉取。AIServer 不从本地保存点开始训练，也不会在正常停止时删除缓存。

## 6. 默认地址

| 服务 | 地址 |
| --- | --- |
| AIServer gRPC | `0.0.0.0:9002` |
| Learner Sample Pool | `maze-learner:9100` |
| Learner Model Distributor | `maze-learner:9200` |

## License

[MIT License](LICENSE)
