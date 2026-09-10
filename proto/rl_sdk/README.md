# C++ Client SDK

训练组维护公共源码，随 `rl-contracts` artifact 的 `sdk/` 交付。环境项目直接链接 `rl_sdk::sdk`，通过 `TaskClient<Protocol>` 调用任务 RPC。环境代码负责初始化、生成 Proto 状态、执行动作和处理任务结果；channel、Stub、deadline、命令序号、WAIT 与回执确认由库持有。

## 依赖与构建

依赖 C++17、Protobuf、gRPC、pthread。生成当前开发制品：

```sh
bash ./build_dev_artifact.sh task-maze
```

任意 CMake 项目可直接消费制品，不依赖 Maze Client 源码：

```cmake
add_subdirectory("${CONTRACT_ARTIFACT}/sdk" rl-sdk)
rl_sdk_add_task(task_protocol "${CONTRACT_ARTIFACT}/cpp" proto/maze/maze)
add_executable(environment_client main.cpp)
target_link_libraries(environment_client PRIVATE task_protocol)
```

`rl_sdk_add_task` 组装公共 communication、identity 和指定任务生成代码，并传递库依赖；消费端无需重复列出 gRPC 生成源码或链接库。仓内开发示例见 `maze-client/CMakeLists.txt`：`maze_environment` 不依赖通信，`maze_client_adapter` 链接环境库和任务协议库。

## 类型绑定与调用

Proto 及其编译产物是 Client 与 AIServer 唯一共同维护的通信合同。每个任务保持自己的强类型 Service、Req / Rsp。`protoc-gen-rl-sdk` 从同一次 Proto 编译的 Service descriptor 自动生成 `<ServiceName>Protocol`，文件名沿用协议名称，如 `maze.proto` 生成 `maze.sdk.pb.h`；不维护手写 `protocol.h` 或任务专属 RPC 调用实现。

新增任务沿用公共 SDK 的生命周期方法、command / reply、assignment / task_complete 和 wait 结构，任务状态、动作、初始化参数和终局结果使用自己的 Proto 类型。生成器只绑定类型，不解释字段业务语义。环境组在本地 Adapter 填充状态、执行动作和 Reset；训练组在 AIServer 的本地 Adapter 编码 Observation、计算 Reward 和生成任务结果。两侧 Adapter 是各自的任务实现，不是第二份跨团队通信合同。

生成器随 SDK 的 `tools/protoc-gen-rl-sdk` 交付，使用已有的 Python Protobuf 编译依赖。
Contracts 的构建入口自动调用它；独立协议工程也可使用
`protoc --plugin=protoc-gen-rl_sdk=/path/to/sdk/tools/protoc-gen-rl-sdk --rl_sdk_out=cpp .../task.proto`，
与同一次构建的 C++ / gRPC 产物一起交付。

```cpp
#include "rl_sdk/task_client.h"
#include "proto/maze/maze.sdk.pb.h"

rl_sdk::ClientOptions options;
options.identity.set_component("maze-client");
options.identity.set_instance_id("environment-worker-0");
options.identity.set_lifecycle_epoch(1);
options.environment_instance_id = "environment-0";
options.open_request_id = "environment-worker-0:run-123:open";
rl_sdk::TaskClient<rl::task::maze::v1::MazeTaskServiceProtocol> client(options);
```

连接后按任务生命周期调用：

| 方法 | 环境侧提供或消费 | SDK 负责 |
| --- | --- | --- |
| `Connect(endpoint)` | AIServer 地址 | channel 与生成 Stub |
| `OpenSession(response, validate)` | 接收并验证任务配置 | Client identity、Open request、Session 绑定 |
| `Init(request, response, validate)` | 环境初始化 Proto | command identity、回执确认 |
| `BeginEpisode(response, validate, stopped)` | Episode assignment、环境 Reset | Begin 调用、Episode cursor |
| `Update(request, response, validate, stopped)` | 当前事实、动作结果与新动作 | exact request、WAIT、序号提交 |
| `EndEpisode(response, validate)` | 最后一次 Update 已确认后接收任务结果 | End 调用、释放 Episode identity |
| `AbortEpisode(request, response)` | 任务 Proto 的中止原因 | 有界 WAIT 与中止回执 |
| `CloseSession(response)` | 应用退出 | Session 关闭确认 |

验证回调只处理任务 payload；公共回执的结果、序号、phase 由 SDK 校验后提交。`cursor()` 只读，调用者不用填充或推进序号。调用返回 `Applied`、`Rejected`、`Unknown`、`Stopped` 或 `WaitExpired`，`error()` 保留失败信息。

`RunEpisode` 接收当前事实、提交、判断终止、执行动作四个回调：初始状态先上报，动作执行后再上报，最后一次终止状态仍须 Update，终局不执行额外动作。`RunSession` 是可选的完整应用驱动；Maze 示例用它组织环境初始化、Episode 循环和退出，Binding 内调用上述公共方法。

## 生命周期与责任边界

每个 `TaskClient` 用于一个同步 Session，不跨线程共享。Request 在一次调用中保持不变，只有 APPLIED / ALREADY_APPLIED 经过确认才推进 cursor。WAIT 不推进环境，也不重新生成事实。Transport 沿用原合同指定 gRPC 状态最多两次尝试及 50 ms 间隔；Update、普通命令和 Abort WAIT 的预算由 `ClientOptions` 声明。Unknown 不转为替代 Abort / Close，不提供跨进程会话恢复。

`session.h`、`transport.h` 是公共实现；环境接入优先使用 `task_client.h`。`server_command.h` / `replay_window.h` 提供服务侧命令提交，`metric_catalog.h` 提供独立指标目录服务。

AIServer 的 `TrainingTaskService<Protocol, Task>` 在训练侧统一七个 RPC 的生命周期和样本事务。Task Adapter 根据环境 Proto 编码 Observation、计算 Reward、映射 Action 与生成任务指标；`TrainingTransaction` 负责 pending transition、GAE 分段、采样动作和样本队列提交。Client SDK 不计算奖励或 GAE，也不连接 Learner、Pool 或 Model Distributor。Learner 的算法与任务 Proto 没有依赖关系。

## Maze 接入位置

- `maze-client/src/maze/`：Maze 协议适配、动作回执、Episode 分配、环境、配置和回放；应用入口在 `main/main.cpp`，通用 RPC / Session 继续直接依赖公共 SDK。
- `rl-aiserver/src/task/`：按 `protocol/`、`config/`、`runtime/`、`session/`、`inference/`、`policy/`、`reward/`、`sample/`、`model/`、`metrics/` 分类管理通用组件，不依赖 Maze 类型或 Proto。
- `rl-aiserver/src/maze/`：Maze 字段适配、Observation、Reward、地图、Episode 控制、任务指标和配置。`task_entry.h` / `sources.cmake` 是本地任务装配与构建清单；使用生成的 RPC 类型，不声明另一套 wire 合同。
- 任务 Proto 和编译产物保留在两端的 `proto/maze/`；具体任务实现调用通用组件，通用组件不反向导入 Maze。
- 公共 `TrainingTaskService` 持有所有任务 RPC 实现。AIServer 可执行程序为 `rl_aiserver`，通用入口不包含 Maze 类型。

当前重新开局沿用 `EndEpisode → BeginEpisode → EpisodeAssignment → Environment.Reset`。
WAIT 保留原请求且不前进环境。当前 Proto 没有中途 Reset 或重启 Client 进程的指令，目录与绑定整理不增加这些操作。
