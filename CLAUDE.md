# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`vision-rs` is a high-performance computer vision SDK for Rust (crate name `vision_rs`, edition 2024, AGPL-3.0). It is in early development — `src/lib.rs` is currently a stub.

## Commands

```bash
cargo build               # Build
cargo test                # Run all tests
cargo test <test_name>    # Run a single test by name
cargo clippy              # Lint
cargo fmt                 # Format
cargo doc --open          # Build and open docs locally
```

## Architecture

The library is a single crate (`src/lib.rs` as the root). As modules are added, they should be declared here and organized under `src/` as either inline modules or separate files.

## Sibling Projects

### `../teenygrad`

The ML/tensor runtime this project builds on. All its crates are pre-patched in `Cargo.toml` via `[patch.crates-io]`, so adding a dependency is straightforward:

```toml
# In [dependencies] — Cargo resolves to the local path automatically
teeny-core = "0.1"
```

Available crates: `teeny-core`, `teeny-fxgraph`, `teeny-kernels`, `teeny-triton`, `teeny-compiler`, `teeny-macros`, `teeny-cuda`, `teeny-cache`, `teeny-data`, `teeny-http`, `teeny-nlp`, `teeny-onnx`, `teeny-torch`.

When these crates are eventually published to crates.io, remove the `[patch.crates-io]` block from `Cargo.toml` and the version specifiers will resolve normally.

### `../rust`

A custom fork of the Rust compiler used to compile GPU kernels. `vision-rs` does not depend on it directly and is built with the stable release compiler. If a GPU kernel fails to compile with an inexplicable error, the root cause may be in this compiler fork rather than in `vision-rs` or `teenygrad`.
