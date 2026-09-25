# Protocol inputs

`maze/maze.proto` and the independent `rl_sdk/` are the task compiler inputs.
Shared identity and Session sources live only in `rl_sdk/proto/`; task/common/
session bindings are generated in the build tree. Production and the registered
tests share the same core library and generated protocol target.

`training/`, `metrics/` and `maze/metrics.pb.*` retain explicitly adopted generated
training and metric inputs. They use the common identity header from the task
generation target. Shared protocols do not import Maze.

Adopt Task and Training inputs separately with `scripts/sync_contract_snapshot.sh /path/to/task-maze-artifact`
and `scripts/sync_training_snapshot.sh /path/to/training-artifact` from the repository root. Both preserve
unchanged file timestamps. Normal builds never synchronize upstream source.
