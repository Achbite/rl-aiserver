# RL AIServer

简体中文 | [English](README.en.md)

AIServer 提供静态模型评测，以及训练中的推理、轨迹组装和样本发送。A3 本地链由开发者分别
启动 Learner、AIServer 与 Client；Framework 不再编排运行时。

## 1. 开发容器、增量构建与测试

```bash
# 宿主机：自动构建或复用独立开发镜像并进入容器
make shell

# 容器内：构建与测试是两个显式入口
./build.sh
bash ./test.sh

# 宿主机也可复用同一容器执行构建
make build
```

开发容器不继承旧 runtime image，使用持久 ccache。`ninja: no work to do.` 表示源码未变化，
不会自动重复运行测试。测试只能在仓库根通过 `bash ./test.sh` 启动；`build.sh`、Docker image
构建和其他 wrapper 不会隐式运行测试。`make shell` 只能在宿主机执行。

## 2. 运行模式

进入开发容器后：

```bash
# 查看实际二进制接受的覆盖项及其 config 字段（不启动服务）
bash ./run.sh --help

# 使用显式 SaveModel.onnx 的确定性评测
bash ./run.sh --config configs/server_config.yaml --workload evaluation \
  --evaluation-model /absolute/path/SaveModel.onnx

# 训练模式；config 提供默认值，CLI 显式覆盖 Learner 地址
bash ./run.sh --config configs/server_config.yaml --workload training \
  --sample-distributor maze-learner:9100 \
  --model-distributor maze-learner:9200
```

评估的最终 config/CLI 值必须指向一个常规、非符号链接的 `SaveModel.onnx` 文件；不会读取
相邻 manifest，也不允许目录入口。`run.sh` 只监督进程和传播退出码，业务参数只由 C++ 的
config/CLI 层解释。

默认 workload 明确配置在 `configs/server_config.yaml` 的 `server.run_mode`；
`--workload` 只是覆盖它。Reward V4 的公式和数值由 `src/ai/maze_reward.cpp` 固定持有，
YAML 中不提供 `reward:` 调参段；若出现 `reward.*`，配置加载会失败关闭。config 只保留
`training_semantics.reward_schema_*` 身份以验证训练语义。

同一运行镜像提供只读模型诊断，用于验证张量合约和有限值推理：

```bash
/opt/rl/aiserver/bin/maze_aiserver \
  --inspect-model /absolute/path/SaveModel.onnx
```

探针只输出模型张量身份，不加载服务配置，也不启动端口。

## 3. Training 缓存

AIServer 只使用 `model.local_train_dir` 下的私有 `cache`，不接收也不理解平台
`task_id/run_id`。同一目录由 `.aiserver.lock` 防止两个 AIServer 并发使用；正常重启会校验并
恢复既有合法 cache，不因目录非空而删除或拒绝它：

```bash
bash ./run.sh --config configs/server_config.yaml --workload training
```

模型始终从本次隔离训练的 Learner Model Distributor 拉取。AIServer 从 Distributor 状态发现内部模型 lineage，首次发现后固定；同一生命周期出现另一 lineage 会失败关闭。AIServer 不从本地保存点开始训练，也不会在正常停止时删除缓存。

## 4. 正式制品与镜像

只有 Level 1/2 通过、用户 Review 并形成 clean savepoint 后，才同步正式 Contracts artifact
并在宿主机执行 `bash build_image.sh`。正式构建要求运行仓 clean，且不读取开发 artifact 或
开发容器 build 目录。

## 5. 默认地址

| 服务 | 地址 |
| --- | --- |
| AIServer gRPC | `0.0.0.0:9002` |
| Learner Sample Pool | `maze-learner:9100` |
| Learner Model Distributor | `maze-learner:9200` |

## License

[MIT License](LICENSE)
