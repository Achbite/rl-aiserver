# SampleDistributor Runtime Artifact

This directory is the staging location for the SampleDistributor artifact
selected for the AIServer image. The artifact is copied here explicitly after
`rl-sample-pool` publishes it to the workspace artifact store.

For the versions in `../artifact_versions.env`, run from `rl-aiserver`:

```bash
cp -R ../.workspace/artifacts/rl-sample-pool/0.3.0/linux-arm64/. \
    sample-distributor/
```

`bin/`, `config/`, and `manifest.json` are generated inputs and are not
committed. `build_image.sh` verifies their package, version, platform,
Contracts identity, and SHA-256 values before building the image; an older
staged artifact is rejected.
