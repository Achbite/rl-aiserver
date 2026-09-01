# Maze Contract Snapshot

This directory owns the task and training protocol inputs used by this AIServer
checkout. AIServer always compiles these repository-local files and never
discovers or mounts an external Contracts artifact.

`manifest.json` describes only the local Maze task protocol snapshot. Run
`../scripts/sync_contract_snapshot.sh` only when you explicitly choose to
replace the Maze task files with the release selected in
`artifact_versions.env`. That operation does not replace `training.proto` or
its generated service bindings. Normal builds and `make shell` never run the
sync command, and Client/AIServer communication is not gated by source,
generator, hash, or platform equality.
