# vision-rs

A high-performance computer vision SDK for Rust.

## Building

### Host (native)

```bash
cargo build --release
```

### Cross-compilation for Jetson Orin Nano

vision-rs targets the Jetson Orin Nano (`aarch64-unknown-linux-gnu`) using
[cross](https://github.com/cross-rs/cross) via the `cargo-teeny` plugin.

#### Prerequisites

1. **Install cross**
   ```bash
   cargo install cross --git https://github.com/cross-rs/cross
   ```

2. **Add the aarch64 target**
   ```bash
   rustup target add aarch64-unknown-linux-gnu
   ```

3. **Install cargo-teeny**
   ```bash
   cargo install --git https://github.com/spinorml/cargo-teeny
   ```

4. **CUDA aarch64 libraries** — the build mounts the host CUDA aarch64 target
   directory into the cross container. The default path is:
   ```
   /usr/local/cuda-12.6/targets/aarch64-linux
   ```
   If your CUDA installation is at a different path, pass `--cuda-path <path>` (see below).

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

# Custom CUDA path
cargo teeny build --target jetson-orin-nano --cuda-path /usr/local/cuda-12.8/targets/aarch64-linux
```

Compiled artifacts land in `target/aarch64-unknown-linux-gnu/release/`.

#### How it works

`cargo teeny build` wraps `cross build` and automatically:

- Resolves the [teenygrad](https://github.com/spinorml/teenygrad) workspace
  root from the `[patch.crates-io]` entries in `Cargo.toml` and mounts it
  into the cross container (required because cross only auto-mounts individual
  crate directories, not the workspace root that provides `Cargo.toml`
  inheritance).
- Mounts the host CUDA aarch64 target directory at the path the cross
  container's Dockerfile expects.
- Uses the custom Docker image defined in `docker/Dockerfile.jetson-orin-nano`,
  which extends the cross base image with clang-12 (required by `bindgen` for
  aarch64 cross-bindings).
