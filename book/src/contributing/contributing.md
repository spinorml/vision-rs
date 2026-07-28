# Contributing to vision-rs

vision-rs is under active development. Before contributing:

- **File an issue first** for anything beyond a trivial fix, so the change
  is discussed before you invest time in it.
- **Open a pull request** against [teenygrad/vision-rs](https://github.com/teenygrad/vision-rs).

## Engineering standards

See the repository's `CLAUDE.md`/`AGENTS.md` for the full conventions this
project follows — in short: prefer clarity over cleverness, avoid
`unwrap()`/`expect()` outside test code, and add tests for behavior
changes. Kernel code (anything under `models::yolo::kernels`) additionally
needs a snapshot test of its generated source/MLIR — see the existing
`tests/test_*.rs` files for the pattern.

## Documentation standards

Public items need `///` doc comments — `cargo doc`/CI enforce this via
`#![warn(missing_docs)]`. If you're adding a new `#[kernel]`-annotated
function, the doc comment on the function is what ends up on the generated
`Kernel` struct (the actual public API surface); write it there.

## Related repositories

- [teenygrad](https://github.com/teenygrad/teenygrad) — the ML runtime
  vision-rs is built on. Changes to shared kernel/graph infrastructure
  (including the `#[kernel]` macro itself) belong there, not in vision-rs.
- [teenygrad/teeny](https://github.com/teenygrad/teeny) — the `teenyc`
  compiler fork used to compile GPU kernels.
