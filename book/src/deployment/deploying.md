# Deploying to the Device

## Copying the package over

```bash
cargo teeny deploy \
  --package ./dist/yolo26-orin \
  --host <user>@<orin-host> \
  --dest /home/<user>/vision-rs-yolo26
```

- Uses `rsync -a` over SSH; set up key-based auth on the device first, or
  it'll prompt for a password interactively (stdio is inherited, so that
  works too).
- By default, re-running `deploy` only copies files that aren't already on
  the remote — safe to re-run after a partial transfer. Pass `--overwrite`
  to force a full re-sync, e.g. after rebuilding/repackaging with changes.
- `--ssh "ssh -p <port>"` if the device uses a non-default SSH port.

## Running on the device

```bash
ssh <user>@<orin-host>
cd /home/<user>/vision-rs-yolo26
```

`data/` was scaffolded empty by `package` — populate it before running
anything that needs a model/dataset, either by:

- letting the binary download it directly on the device (if it has
  internet access):
  ```bash
  ./bin/yolo26 download --dataset assets/datasets/coco128.toml
  ```
- or `rsync`-ing pre-downloaded models/datasets from your host's
  `$MODELS_CACHE_DIR`/`$DATASETS_CACHE_DIR` into `data/` on the device
  instead.

Then run a smoke test, e.g. the same throughput/latency benchmark used in
development (see [Benchmarking & Profiling](../kernels-and-performance/benchmarking.md)):

```bash
./bin/yolo26 bench \
  --model ultralytics/yolo26n \
  --dataset assets/datasets/coco128.toml \
  --skip-map \
  --warmup 10 \
  --runs 100
```

See the project's `CLAUDE.md` for the full set of `yolo26` subcommands
(`view`, `verify`, `train`, `validate`, ...).
