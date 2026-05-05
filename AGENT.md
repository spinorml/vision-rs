# AGENT.md — Rust Best Practices

## Ownership & Borrowing

- Prefer borrowing (`&T`, `&mut T`) over cloning unless ownership transfer is semantically correct.
- Avoid unnecessary `clone()` — if you reach for it, reconsider the data flow.
- Use `Cow<'_, T>` when a function sometimes needs owned data and sometimes can borrow.
- Return owned types from public APIs; accept borrowed types as arguments (`&str` not `String`, `&[T]` not `Vec<T>`).

## Error Handling

- Use `thiserror` for library error types — derive `Error` on an enum that covers each failure mode explicitly.
- Never use `unwrap()` or `expect()` in library code (only in tests and examples).
- Propagate errors with `?`; only convert at API boundaries where the caller needs a different type.
- Avoid `Box<dyn Error>` in public APIs — callers can't match on it.

## Performance

- Prefer iterators over index-based loops; they enable bound-check elision and better auto-vectorization.
- Annotate hot paths with `#[inline]` and cold error paths with `#[cold]`.
- Use `#[repr(C)]` or `#[repr(transparent)]` on types that cross FFI or SIMD boundaries.
- Avoid heap allocation in inner loops — preallocate buffers and pass them in.
- Profile before optimizing: `cargo flamegraph` or `perf` first.

## Safety & Unsafe

- Minimize `unsafe` surface — each `unsafe` block must have a `// SAFETY:` comment explaining the invariant upheld.
- Encapsulate `unsafe` inside safe abstractions; never let unsafe preconditions leak to callers.
- Run `cargo miri test` to catch UB in unsafe code.

## API Design

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).
- Use the builder pattern for types with many optional parameters.
- Implement standard traits where applicable: `Debug`, `Clone`, `PartialEq`, `Display`, `From`/`Into`, `Default`.
- Seal traits that are not intended for downstream implementation (`mod private { pub trait Sealed {} }`).
- Mark experimental or unstable items with `#[doc(hidden)]` or a feature flag, not just a comment.

## Testing

- Unit tests live in a `#[cfg(test)]` module at the bottom of the file they test.
- Integration tests go in `tests/` and exercise the public API only.
- Use `#[should_panic(expected = "...")]` sparingly — prefer `Result`-returning tests.
- Benchmark with `criterion` — add benches under `benches/`.

## Clippy & Formatting

- Code must pass `cargo clippy -- -D warnings` with no suppressions unless justified.
- Suppress a lint with `#[allow(clippy::lint_name)]` at the narrowest scope, always with a comment explaining why.
- `cargo fmt` is non-negotiable — CI should enforce it.

## Dependencies

- Add dependencies deliberately — evaluate maintenance status and compile-time cost.
- Prefer `no_std`-compatible crates where possible to keep the library portable.
- Specify minimum required versions, not `*`.

## Kernel Compiler Guardrails

- For BatchNorm kernels targeting Triton, **never place `T::sum` (or other reductions) inside `scf.for` loops**.
- Triton currently cannot lower reductions nested in `scf.for`; keep reductions outside the loop structure.
- If a refactor reintroduces an in-loop reduction, treat it as a compiler-compatibility bug and revert to the "reduce outside loop" pattern.
