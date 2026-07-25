# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`vision-rs` is a high-performance computer vision SDK for Rust (crate name `vision_rs`, edition 2024, Apache-2.0). It is in early development — `src/lib.rs` is currently a stub.

## Commands

```bash
cargo build               # Build
cargo test                # Run all tests
cargo test <test_name>    # Run a single test by name
cargo clippy              # Lint
cargo fmt                 # Format
cargo doc --open          # Build and open docs locally
```

## Benchmarking

### Build (required before any bench run)

```bash
source .env
cargo build --release --example yolo26 --features cuda
```

### Run benchmark (clean numbers)

```bash
source .env
./target/release/examples/yolo26 bench \
  --model ultralytics/yolo26n \
  --dataset assets/datasets/coco128.toml \
  --skip-map \
  --warmup 10 \
  --runs 100
```

Remove `--skip-map` to also compute mAP@0.5 (adds ~30s).

### nsys profiling (per-kernel breakdown)

The bench command wraps the batch=1 timed loop with `cudaProfilerStart/Stop`, so
`--capture-range=cudaProfilerApi` records only the timed region (not warmup or
compile). Use `--cuda-graph-trace=node` to get individual kernel stats inside CUDA
graphs, which are otherwise opaque to nsys.

```bash
source .env
nsys profile \
  --capture-range=cudaProfilerApi \
  --cuda-graph-trace=node \
  --output=/tmp/yolo26_bench \
  --force-overwrite=true \
  --stats=true \
  --kill=none \
  ./target/release/examples/yolo26 bench \
    --model ultralytics/yolo26n \
    --dataset assets/datasets/coco128.toml \
    --skip-map \
    --warmup 10 \
    --runs 100
```

`--kill=none` prevents nsys from SIGTERM-ing the process after the capture range
ends (without it the final results row never prints).

The report is saved to `/tmp/yolo26_bench.nsys-rep` and can be opened in the
Nsight Systems GUI. The `--stats=true` flag prints `cuda_gpu_kern_sum`,
`cuda_api_sum`, and `cuda_gpu_mem_time_sum` tables directly to stdout.

### TensorRT baseline (bench.py)

Runs the ultralytics benchmark across PyTorch / ONNX Runtime / TensorRT FP32.
TRT engine export takes ~25s on first run; the `.engine` file is cached at
`$MODELS_CACHE_DIR/ultralytics/yolo26n/yolo26n.engine`.

```bash
source .env
python3 bench.py
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


<!-- BEGIN BEADS INTEGRATION v:2 profile:minimal (br) -->
## Beads Issue Tracker

This project uses **br (beads_rust)** for issue tracking.

**Note:** `br` is non-invasive and never executes git commands. After `br sync --flush-only`, you must manually run `git add .beads/ && git commit`.

### Quick Reference

```bash
br ready              # Find available work
br show <id>          # View issue details
br update <id> --claim  # Claim work
br close <id>         # Complete work
```

### Rules

- Use `br` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Use `br` comments/labels for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   br sync --flush-only
   git add .beads/
   git commit -m "sync beads"
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
