# vision-rs

A high-performance computer vision SDK for Rust.

## Prerequisites

- **OS**: Ubuntu. This is the only supported development/build platform.
- **Rust**: the latest stable toolchain, via [rustup](https://rustup.rs/):
  ```bash
  rustup toolchain install stable
  rustup default stable
  ```
- **CUDA Toolkit 13.3 or later** on the host (for the `cuda` feature — on by default — and for
  ahead-of-time kernel compilation via the custom `teenyc` compiler; see below). Check with:
  ```bash
  nvcc --version
  ```
- **This crate's sibling repos**, checked out next to `vision-rs` (its `Cargo.toml` patches
  `teeny-*` crates to `../teenygrad/...` and cross builds auto-mount that workspace root):
  ```bash
  git clone https://github.com/spinorml/teenygrad ../teenygrad
  ```
- Only if you'll cross-compile/deploy to a Jetson Orin Nano (see below): `cross`, Docker, the
  `aarch64-unknown-linux-gnu` rustup target, `cargo-teeny`, and SSH/`rsync` access to the device.

## Getting started

```bash
git clone https://github.com/spinorml/vision-rs
cd vision-rs
cp .env.dev .env   # then edit the paths inside for your machine — see below
```

## Installing the toolchain

vision-rs builds against the **stable** Rust compiler, but GPU kernels (Triton/MLIR) are compiled
— both ahead-of-time (`cargo teeny package`/`aot`) and at runtime via JIT — by a separate, custom
compiler fork: `teenyc` (see [teenygrad/teeny](https://github.com/teenygrad/teeny)). You need that binary on your machine and
`TEENYC_PATH` (in `.env`) pointing at it before building/running anything that uses the `cuda`
feature (on by default).

The supported way to get `teenyc` is via [`cargo-teeny`](https://github.com/spinorml/cargo-teeny),
which installs a prebuilt release from the spinorml CDN — you do **not** need to clone or build
the `teeny` compiler fork from source for this.

1. **Install `cargo-teeny`**:
   ```bash
   cargo install --git https://github.com/spinorml/cargo-teeny
   ```

2. **Install the `teenyc` toolchain**:
   ```bash
   cargo teeny install-toolchain
   ```
   This downloads and `rustup toolchain link`s the compiler as
   `stable-teenyc-x86_64-unknown-linux-gnu` under `~/.rustup/toolchains/`. Verify it:
   ```bash
   rustup toolchain list | grep teeny
   rustup run stable-teenyc-x86_64-unknown-linux-gnu teenyc --version
   ```

3. **Point `TEENYC_PATH` at it.** `.env.dev` (copied to `.env` in "Getting started" above) already
   defaults to the right path:
   ```bash
   export TEENYC_PATH=$HOME/.rustup/toolchains/stable-teenyc-x86_64-unknown-linux-gnu/bin/teenyc
   ```
   Adjust `DATASETS_CACHE_DIR`/`MODELS_CACHE_DIR` in the same file to wherever you want benchmark
   datasets/models cached, then:
   ```bash
   source .env
   ```
   Do this in every shell before building/running/testing/benchmarking — the examples and test
   suite compile kernels via `teenyc` at runtime and need `TEENYC_PATH` set.

## Building

### Host (native)

```bash
source .env
cargo build --release
cargo test
```

### Cross-compilation for Jetson Orin Nano

vision-rs targets the Jetson Orin Nano (`aarch64-unknown-linux-gnu`) using
[cross](https://github.com/cross-rs/cross) via the `cargo-teeny` plugin.

#### Additional prerequisites

1. **Install cross**
   ```bash
   cargo install cross --git https://github.com/cross-rs/cross
   ```

2. **Add the aarch64 target**
   ```bash
   rustup target add aarch64-unknown-linux-gnu
   ```

3. **CUDA aarch64 libraries** — the build mounts the host CUDA aarch64 target directory into the
   cross container so the cross-compiled binary links against the CUDA version actually present on
   the Jetson Orin Nano's JetPack (independent of the host's own CUDA 13.3 from the prerequisites
   above, which is only used for AOT-compiling kernels on the host). The default path is:
   ```
   /usr/local/cuda-12.6/targets/aarch64-linux
   ```
   If your target device's JetPack CUDA version differs, pass `--cuda-path <path>` on every
   `build`/`package` command below.

#### Build

```bash
# Build the library in release mode (default)
cargo teeny build --target jetson-orin-nano

# Build all examples
cargo teeny build --target jetson-orin-nano --examples

# Build a single example
cargo teeny build --target jetson-orin-nano --example yolo26

# Type-check only (faster feedback)
cargo teeny check --target jetson-orin-nano

# Lint
cargo teeny clippy --target jetson-orin-nano

# Debug build
cargo teeny build --target jetson-orin-nano --no-release

# Custom CUDA path (target device's JetPack CUDA version)
cargo teeny build --target jetson-orin-nano --cuda-path /usr/local/cuda-12.8/targets/aarch64-linux
```

Compiled artifacts land in `target/aarch64-unknown-linux-gnu/release/`.

#### How it works

`cargo teeny build` wraps `cross build` and automatically:

- Resolves the [teenygrad](https://github.com/spinorml/teenygrad) workspace root from the
  `[patch.crates-io]` entries in `Cargo.toml` and mounts it into the cross container (required
  because cross only auto-mounts individual crate directories, not the workspace root that
  provides `Cargo.toml` inheritance).
- Mounts the host CUDA aarch64 target directory at the path the cross container's Dockerfile
  expects.
- Uses the custom Docker image defined in `docker/Dockerfile.jetson-orin-nano`, which extends the
  cross base image with clang-12 (required by `bindgen` for aarch64 cross-bindings).

## Packaging a deployable bundle

`cargo teeny package` combines cross-compiling the binary/example for the target board with
ahead-of-time-compiling its GPU kernels on the host, into one self-contained directory you can
copy straight to the device — no `teenyc`, CUDA toolkit, or Rust install needed on the Jetson
itself.

```bash
cargo teeny package \
  --target jetson-orin-nano \
  --example yolo26 \
  --dest ./dist/yolo26-orin \
  --device cuda \
  --options "capability=sm_87,ptx-version=82"
```

- `--options capability=sm_87` is the Jetson Orin Nano's GPU compute capability (Ampere).
  `ptx-version=82` overrides `teenyc`'s otherwise-conservative default PTX ISA floor for `sm_87` —
  keep this pinned unless your target device is on a materially different CUDA version.
- Use `--bin <name>` instead of `--example <name>` when packaging a binary crate.
- This produces:
  ```
  dist/yolo26-orin/
    bin/yolo26        # cross-compiled binary
    cache/            # AOT-compiled GPU kernels
    conf/             # provenance marker (target/device/options/commit/build time)
    data/             # empty — populate with models/datasets separately (see "Running", below)
  ```
  The binary auto-detects `cache/` as its sibling directory at runtime (no extra env var needed
  on the device — see `teeny_compiler::compiler::default_cache_dir()`), so it uses the
  pre-compiled kernels instead of trying to JIT-compile (which would fail: there's no `teenyc` on
  the Jetson).

## Deploying to the Orin

```bash
cargo teeny deploy \
  --package ./dist/yolo26-orin \
  --host <user>@<orin-host> \
  --dest /home/<user>/vision-rs-yolo26
```

- Uses `rsync -a` over SSH; set up key-based auth on the device first, or it'll prompt for a
  password interactively (stdio is inherited, so that works fine too).
- By default, re-running `deploy` only copies files that aren't already on the remote (safe to
  re-run after a partial transfer). Pass `--overwrite` to force everything to re-sync, e.g. after
  rebuilding/repackaging with changes.
- `--ssh "ssh -p <port>"` if the device uses a non-default SSH port.

## Running on the device

```bash
ssh <user>@<orin-host>
cd /home/<user>/vision-rs-yolo26
```

`data/` was scaffolded empty by `package` — populate it before running anything that needs a
model/dataset, either by:

- letting the binary download it directly on the device (if it has internet access), e.g.:
  ```bash
  ./bin/yolo26 download --dataset assets/datasets/coco128.toml
  ```
- or `rsync`-ing pre-downloaded models/datasets from your host's `$MODELS_CACHE_DIR`/
  `$DATASETS_CACHE_DIR` (see `.env`) into `data/` on the device instead.

Then run a smoke test, e.g. the same throughput/latency benchmark used in development:

```bash
./bin/yolo26 bench \
  --model ultralytics/yolo26n \
  --dataset assets/datasets/coco128.toml \
  --skip-map \
  --warmup 10 \
  --runs 100
```

See `CLAUDE.md` for the full set of `yolo26` subcommands (`view`, `verify`, `train`, `validate`,
...) and additional benchmarking/profiling workflows.
