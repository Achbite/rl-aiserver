# RL-SDK

训练组维护公共源码，随 `rl-contracts` artifact 的 `sdk/` 交付。环境项目直接链接 `rl_sdk::sdk`，通过 `TaskClient<Protocol>` 调用任务 RPC。环境代码负责初始化、生成 Proto 状态、执行动作和处理任务结果；channel、Stub、deadline、命令序号、WAIT 与回执确认由库持有。

## 依赖与构建

依赖 C++17、Protobuf、gRPC、pthread；本地代码生成使用 Python 3.9 或更高版本及其 Protobuf 包、`protoc` 与 `grpc_cpp_plugin`。`generate-task` 和它启动的 SDK 插件使用同一个 Python 解释器；通过 CMake 调用时沿用 `Python3_EXECUTABLE`，Protobuf 包须安装在该解释器环境中。从 Contracts 仓库导出独立源码包：

```sh
bash sdk/build_artifact.sh /path/to/output
# 输出 /path/to/output/RL-SDK.tar.gz，解压后顶层目录为 RL-SDK/
```

包内包含 SDK headers、CMake 入口、生成器、公共 `proto/common/identity.proto` 和 `proto/communication/session.proto`。这两份公共 Proto 源由 Contracts 维护，打包时复制；任务 Proto 由任务双方约定并各自放进自己的项目。新增任务无需修改 Contracts 的任务 profile，也不需要在本地安装 Maze Client。

Client 与 AIServer 可以分别从相同的任务 Proto，在各自的编译环境生成 `.pb.cc/.pb.h`、`.grpc.pb.cc/.grpc.pb.h` 与 `.sdk.pb.h`：

```cmake
add_subdirectory("${RL_SDK_DIR}" rl-sdk)
rl_sdk_generate_task(task_protocol
    PROTO proto/my_task/task.proto
    IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")
add_executable(environment_client main.cpp)
target_link_libraries(environment_client PRIVATE task_protocol)
```

`PROTO` 是相对 import root 的协议路径，也可以是某个 import root 内的绝对路径。`IMPORT_DIRS` 可指定多个任务依赖目录。生成器使用 Protobuf descriptor 解析 import 闭包，自动生成和链接任务导入的消息，并跟踪这些源码的构建依赖；Protobuf well-known types 由 `libprotobuf` 提供。输出只进入构建目录，不改写项目保存的 Proto 或生成快照。

也可以在任意构建系统中直接调用：

```sh
python3 /path/to/RL-SDK/tools/generate-task \
  --task-proto proto/my_task/task.proto \
  --import-dir /path/to/environment-project \
  --output-dir /path/to/build/generated
```

`rl_sdk_add_task(target protocol_root task_stem)` 仍用于显式链接已经生成的协议产物。`build_dev_artifact.sh task-maze` 是本仓现有任务的预生成打包入口，独立任务不依赖它。仓内实际接入见 `maze-client/CMakeLists.txt`：`maze_environment` 不依赖通信，`maze_client_adapter` 链接环境库和通过 `rl_sdk_generate_task` 在本地生成的任务协议库。

## 类型绑定与调用

Proto 及其编译产物是 Client 与 AIServer 唯一共同维护的通信合同。每个任务保持自己的强类型 Service、Req / Rsp。`protoc-gen-rl-sdk` 从同一次 Proto 编译的 Service descriptor 自动生成 `<ServiceName>Protocol`，文件名沿用协议名称，如 `maze.proto` 生成 `maze.sdk.pb.h`；不维护手写 `protocol.h` 或任务专属 RPC 调用实现。

新增任务沿用公共 SDK 的生命周期方法、command / reply、assignment / task_complete 和 wait 结构，任务状态、动作、初始化参数和终局结果使用自己的 Proto 类型。生成器只绑定类型，不解释字段业务语义。环境组在本地 Adapter 填充状态、执行动作和 Reset；训练组在 AIServer 的本地 Adapter 编码 Observation、计算 Reward 和生成任务结果。两侧 Adapter 是各自的任务实现，不是第二份跨团队通信合同。

生成器随 SDK 的 `tools/protoc-gen-rl-sdk` 交付，使用已有的 Python Protobuf 编译依赖。
Contracts 的构建入口自动调用它；独立协议工程也可使用
`protoc --plugin=protoc-gen-rl_sdk=/path/to/sdk/tools/protoc-gen-rl-sdk --rl_sdk_out=cpp .../task.proto`，
与同一次构建的 C++ / gRPC 产物一起交付。
直接调用上述 protoc 插件时，PATH 中的 `python3` 需满足前述 Python 与 Protobuf 依赖。

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

`RunEpisode` 接收当前事实、提交、判断终止、执行动作四个回调：初始状态先上报，动作执行后再上报，最后一次终止状态仍须 Update，终局不执行额外动作。`RunSession` 是可选的完整应用驱动；Maze 示例用它组织环境初始化、Episode 循环和退出，Binding 内调用上述公共方法。已失败的操作不会被后续 Abort / Close 的结果覆盖；SDK 的 `error()` 保留原始失败，并在清理也失败时追加该清理错误。Transport 错误保留 gRPC 状态码与原始消息，任务 Adapter 不应读取失败 RPC 的空 response 来替代它。

## 生命周期与责任边界

每个 `TaskClient` 用于一个同步 Session，不跨线程共享。Request 在一次调用中保持不变，只有 APPLIED / ALREADY_APPLIED 经过确认才推进 cursor。WAIT 不推进环境，也不重新生成事实。Transport 沿用原合同指定 gRPC 状态最多两次尝试及 50 ms 间隔；Update、普通命令和 Abort WAIT 的预算由 `ClientOptions` 声明。Unknown 不转为替代 Abort / Close，不提供跨进程会话恢复。

`session.h`、`transport.h` 是公共实现；环境接入优先使用 `task_client.h`。`server_command.h` / `replay_window.h` 提供服务侧命令提交。

`metric_catalog.h` 是可选的指标目录服务 helper，任务通信不依赖它。使用时需要另行取得同一 Contracts 的 `proto/metrics/catalog.proto` 与 `proto/metrics/registry.proto`，生成并链接 `catalog.pb.cc`、`catalog.grpc.pb.cc` 和 `registry.pb.cc` 及其头文件。公共 identity 复用已经生成的协议库，不重复链接。独立任务 SDK 源码包不包含这些指标 Proto。

AIServer 的 `TrainingTaskService<Protocol, Task>` 在训练侧统一七个 RPC 的生命周期和样本事务。Task Adapter 根据环境 Proto 编码 Observation、计算 Reward、映射 Action 与生成任务指标；`TrainingTransaction` 负责 pending transition、GAE 分段、采样动作和样本队列提交。Client SDK 不计算奖励或 GAE，也不连接 Learner、Pool 或 Model Distributor。Learner 的算法与任务 Proto 没有依赖关系。

## Maze 接入位置

- `maze-client/src/maze/`：Maze 协议适配、动作回执、Episode 分配、环境、配置和回放；应用入口在 `main/main.cpp`，通用 RPC / Session 继续直接依赖公共 SDK。
- `rl-aiserver/src/task/`：按 `protocol/`、`config/`、`runtime/`、`session/`、`inference/`、`policy/`、`reward/`、`sample/`、`model/`、`metrics/` 分类管理通用组件，不依赖 Maze 类型或 Proto。
- `rl-aiserver/src/maze/`：Maze 字段适配、Observation、Reward、地图、Episode 控制、任务指标和配置。`task_entry.h` / `sources.cmake` 是本地任务装配与构建清单；使用生成的 RPC 类型，不声明另一套 wire 合同。
- 任务 Proto 和编译产物保留在两端的 `proto/maze/`；具体任务实现调用通用组件，通用组件不反向导入 Maze。
- 公共 `TrainingTaskService` 持有所有任务 RPC 实现。AIServer 可执行程序为 `rl_aiserver`，通用入口不包含 Maze 类型。

当前重新开局沿用 `EndEpisode → BeginEpisode → EpisodeAssignment → Environment.Reset`。
WAIT 保留原请求且不前进环境。当前 Proto 没有中途 Reset 或重启 Client 进程的指令，目录与绑定整理不增加这些操作。
