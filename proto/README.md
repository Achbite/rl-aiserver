# Protocol Snapshots

This directory owns the task and training protocol inputs used by this AIServer
checkout. AIServer always compiles these repository-local files and never
discovers or mounts an external Contracts artifact.

Sources and generated `.pb.*` / `.grpc.pb.*` files share categorized paths:

| Directory | Contents |
| --- | --- |
| `common/` | Component identity. |
| `communication/` | Shared Session and command lifecycle. |
| `training/` | Model identity and task-neutral training services. |
| `metrics/` | Registry, catalog and opaque event transport. |
| `tasks/maze/` | Maze typed RPC and Maze Episode calculation facts. |
| `rl_sdk/` | SDK header snapshot supplied by Contracts. |

The repository root is the include root for `proto/...` generated headers.
Shared protocols do not import `tasks/maze`; the task and training snapshots are
combined only at AIServer assembly. The flat generated files have been replaced.

Run `bash ../scripts/sync_contract_snapshot.sh` only when you explicitly choose
to replace the Maze task protocol files with the current `rl-contracts` checkout.
Run `bash ../scripts/sync_training_snapshot.sh` separately when adopting its
task-neutral Training Proto. Normal builds, `make shell`, and runtime-artifact
synchronization never run either command. Client/AIServer communication is
determined by their protobuf wire fields, not a protocol ID, source hash, package
version, generator, or platform identity.
