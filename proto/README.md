# Maze Contract Snapshot

This directory is the staging location for the selected `rl-contracts 0.4.0`
Maze protocol and its generated C++ bindings. AIServer compiles these
repository-local files directly and does not mount or discover an external
Contracts artifact.

`manifest.json` records the release identity and SHA-256 values used by the
image build gate. Copy the central artifact here explicitly before building
an image; the build does not update this snapshot and rejects an older staged
version.
