# Maze Contract Snapshot

This directory is the staging location for the selected `rl-contracts 0.10.0`
training and Maze protocols and their generated C++ bindings. AIServer compiles
these repository-local files directly and does not mount or discover an
external Contracts artifact.

`manifest.json` records the release identity and SHA-256 values used by the
image build gate. Run `../scripts/sync_contract_snapshot.sh` to synchronize the
explicit version and platform selected in `artifact_versions.env`. The build
does not update this snapshot and rejects any byte that differs from the
selected artifact.
