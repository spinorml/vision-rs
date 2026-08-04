# Packaging a Deployable Bundle

`cargo teeny package` combines cross-compiling the binary/example for the
target board with ahead-of-time-compiling its GPU kernels on the host, into
one self-contained directory you can copy straight to the device — no
`teenyc`, CUDA toolkit, or Rust install needed on the Jetson itself (see
[The `teenyc` Toolchain](../kernels-and-performance/teenyc-toolchain.md) for
why AOT compilation is what makes this possible).

```bash
cargo teeny package \
  --target jetson-orin-nano \
  --example yolo26 \
  --dest ./dist/yolo26-orin \
  --device cuda \
  --options "capability=sm_87,ptx-version=82,sm-count=8"
```

- `--options capability=sm_87` is the Jetson Orin Nano's GPU compute
  capability (Ampere). `ptx-version=82` overrides `teenyc`'s otherwise
  conservative default PTX ISA floor for `sm_87` — keep this pinned unless
  your target device is on a materially different CUDA version.
- `sm-count=8` is the Jetson Orin Nano's actual SM count (1024 CUDA cores /
  128 per SM) — enables shape-adaptive conv kernel tile-size selection, so
  deep/small-spatial layers pick a smaller tile size (more thread blocks)
  instead of under-occupying this GPU's 8 SMs at the default tile size.
  Omit it to keep the previous fixed-tile-size behavior.
- Use `--bin <name>` instead of `--example <name>` when packaging a binary
  crate rather than an example.

This produces:

```text
dist/yolo26-orin/
  bin/yolo26        # cross-compiled binary
  cache/            # AOT-compiled GPU kernels
  conf/             # provenance marker (target/device/options/commit/build time)
  data/             # empty — populate with models/datasets separately
```

The binary auto-detects `cache/` as its sibling directory at runtime (no
extra env var needed on the device — see
`teeny_compiler::compiler::default_cache_dir()`), so it uses the
precompiled kernels instead of trying to JIT-compile, which would fail:
there's no `teenyc` on the Jetson.

Next: [Deploying to the Device](./deploying.md).
