# RL AIServer

简体中文 | [English](README.en.md)

AIServer 提供静态模型评测，以及训练中的推理、per-Agent R-PIN segment、GAE/Value Target 和
processed-transition 样本发送。本地训练在 Learner ready 后启动 AIServer，再连接 Client。

本地训练只运行 Learner、AIServer 和 Client 三个容器。`make shell` 是宿主机命令：没有
`aiserver-dev` 时构建开发镜像、创建并启动容器；容器存在时只在必要时启动它，然后直接进入。
它不会读取 Contracts 仓，也不会同步或覆盖本仓 `proto/`。完整三容器工作区包含以下同级目录：

```text
workspace/
  rl-contracts/
  rl-sample-pool/
  rl-model-distributor/
  rl-learner/
  rl-aiserver/
  maze-client/
```

前三个仓库不增加运行容器。Sample Pool/Model Distributor 只装配到 Learner；`rl-contracts` 的
Maze Task Proto 只有在开发者显式执行协议同步时才会更新 Client/AIServer。完整启动顺序参阅
[rl-framework](https://github.com/Achbite/rl-framework)。

## 1. 开发容器、增量构建与测试

```bash
# 宿主机：容器不存在时构建并创建；存在时直接复用并进入
make shell

# 容器内：构建与测试是两个显式入口
./build.sh
bash ./test.sh

# 宿主机也可复用同一容器执行构建
make build

# Dockerfile.dev、工具链、环境变量或挂载变化后显式刷新
make dev-refresh
```

开发容器不继承旧 runtime image，使用持久 ccache。`ninja: no work to do.` 表示源码未变化，
不会自动重复运行测试。测试只能在仓库根通过 `bash ./test.sh` 启动；`build.sh`、Docker image
构建和其他 wrapper 不会隐式运行测试。`make shell` 只能在宿主机执行。`make dev-image` 只重建
镜像，不替换已有容器；`make dev-refresh` 才会重建镜像并重建容器。若 AIServer、测试或编译仍在
运行，刷新会明确失败，必须先停止对应进程。上述入口都不会同步协议。

## 2. 运行模式

确认 Learner 已启动后，打开第二个宿主终端：

```bash
# 宿主机
cd /path/to/workspace/rl-aiserver
make shell

# 以下命令在 AIServer 容器内执行
./build.sh

# 查看实际二进制接受的覆盖项及其 config 字段（不启动服务）
bash ./run.sh --help

# 使用显式 ONNX 模型文件进行确定性评测
bash ./run.sh --config configs/server_config.yaml --workload evaluation \
  --evaluation-model /absolute/path/model.onnx

# 训练模式；config 提供默认值，CLI 显式覆盖 Learner 地址
bash ./run.sh --config configs/server_config.yaml --workload training \
  --sample-distributor maze-learner:9100 \
  --model-distributor maze-learner:9200
```

评估的最终 config/CLI 值必须指向一个非空、常规且非符号链接的 ONNX 文件；文件名和上层目录
不属于 AIServer↔Client 或模型分发合同。评测入口不会读取相邻 manifest，也不允许目录入口。
`run.sh` 只监督进程和传播退出码，业务参数只由 C++ 的 config/CLI 层解释。

默认 workload 明确配置在 `configs/server_config.yaml` 的 `server.run_mode`；
`--workload` 只是覆盖它。Reward 公式和数值由 `src/ai/maze_reward.cpp` 固定持有，
YAML 中不提供 `reward:` 调参段；若出现 `reward.*`，配置加载会失败关闭。训练合同及其
observation/action/reward schema 身份只从 `contract.training_contract_path` 指向的制品读取。

Training 使用模型 manifest 中的 `RolloutEstimatorProfile`。AIServer 为每个 Agent 独立 pin
behavior model，默认最多累计 128 条 completed transition 后封口，计算未归一化 GAE/Value Target，
再由进程内 SampleDistributor 批量提交。实际 Agent 数唯一来自
`environment.agent_count/RL_AISERVER_AGENT_COUNT`；`server.max_agents` 只是容量上限。Client、
任务配置、Learner 和 SamplePool 不提供第二个 Agent 数权威入口。

Action mask 由 Training Contract 与 `OpenSessionRsp.environment.action_mask_mode` 共同明确为
`disabled` 或 `required`。关闭时 Client state 和训练样本不带 mask；开启时 AIServer 校验动作维度、
屏蔽不可用 logits，并把同一 mask 随 transition 交给 Learner。它不作为默认必选能力。

segment 连续性、close reason、终止语义和 bootstrap 由 AIServer 在封口时一次性校验；
这些 producer 内部事实不会进入 `ProcessedTransition`。Learner 和 SamplePool 只接收已经计算好的
PPO observation/action/log-probability/value/advantage/value-target。已 Prepare/ACK 的新模型只在
每个 Agent 自己的下一个 segment 激活。

同一运行镜像提供只读模型诊断，用于验证张量合约和有限值推理：

```bash
/opt/rl/aiserver/bin/maze_aiserver \
  --inspect-model /absolute/path/model.onnx
```

探针只输出模型张量身份，不加载服务配置，也不启动端口。

## 3. Training 缓存

AIServer 只使用 `model.local_train_dir` 下的私有 `cache`，不接收也不理解平台
`task_id/run_id`。同一目录由 `.aiserver.lock` 防止两个 AIServer 并发使用。启动时不扫描、恢复或
回填历史 cache；只有当前进程从 Distributor 请求并成功校验的模型 step 才进入内存索引和淘汰范围。
若该 step 的目标目录已经存在，只接受与本次 protobuf manifest 完全一致的内容；其他既有目录既不
作为启动事实读取，也不由当前进程迁移或删除：

```bash
bash ./run.sh --config configs/server_config.yaml --workload training
```

模型始终从本次隔离训练的 Learner Model Distributor 拉取。AIServer 从 Distributor 状态发现内部模型 lineage，首次发现后固定；同一生命周期出现另一 lineage 会失败关闭。AIServer 不从本地保存点开始训练，也不会在正常停止时删除缓存。

## 4. 构建运行镜像

运行镜像直接构建当前工作树和仓库本地 `proto/`。普通构建不读取 Contracts 仓，也不比较
Client/AIServer 的源码、生成器、哈希或平台。只有明确决定采用 Contracts 仓当前 Maze release 时，
才从 Framework 执行以下命令并审查本仓 diff：

```bash
(cd ../rl-framework && bash sync_maze_protocol.sh)
```

随后在宿主机用项目 tag 构建当前源码：

```bash
RL_PROJECT_IMAGE_TAG=maze-tag-001 bash build_image.sh
```

正式构建不读取开发 artifact 或开发容器 build 目录。AIServer 仍按本仓 Training Contract 字段与
digest 校验模型/训练张量语义，但不会用中央 manifest 的包名、平台或跨仓哈希锁定 Client 通信。
完整镜像引用为 `rl-training/aiserver:maze-tag-001`；同名 tag 允许由后续微调构建直接覆盖。

## 5. 刷新与清理开发容器

```bash
make dev-refresh
make dev-clean
```

`dev-refresh` 保留源码与 ccache volume，只替换开发镜像/容器环境；`dev-clean` 删除开发容器。
两者都不会同步协议。

## 6. 默认地址

| 服务 | 地址 |
| --- | --- |
| AIServer gRPC | `0.0.0.0:9002` |
| Learner Sample Pool | `maze-learner:9100` |
| Learner Model Distributor | `maze-learner:9200` |

## License

[MIT License](LICENSE)
