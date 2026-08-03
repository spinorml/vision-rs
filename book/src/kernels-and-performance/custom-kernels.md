# Custom Kernels

vision-rs writes several GPU kernels directly, on top of teenygrad's
Triton-style DSL (see
[Writing a Triton Kernel](https://docs.teenygrad.org/teenygrad/kernels-and-backends/writing-a-kernel)
and [The `#[kernel]` Macro](https://docs.teenygrad.org/teenygrad/kernels-and-backends/kernel-macro)
in the teenygrad book for the underlying mechanics), rather than composing
them purely from teenygrad's built-in ops. All of them live under
`vision_rs::models::yolo::kernels`.

## Flash Attention 2

`kernels::attention::flash_attn2` implements the standard Flash Attention 2
algorithm: online softmax over `[BATCH*N_HEADS, N_CTX, HEAD_DIM]`-layout
tensors, one CTA per `(batch_head, query_row)` pair, so the full
`N_CTX_Q × N_CTX_K` attention matrix is never materialised (memory is
`O(N_CTX × HEAD_DIM)` per CTA rather than `O(N_CTX²)`). `HEAD_DIM` is a
compile-time const generic and must be a power of two. Forward, `dQ`
backward, and `dK`/`dV` backward are separate kernels (`#[kernel]`-annotated
fns), each with their own grid shape.

## Position-Sensitive Attention (PSA)

`kernels::attention::psa` doesn't reimplement attention — it wraps Flash
Attention 2 with the data-rearrangement kernels the YOLO26 `PSABlock` needs
around it:

- **`PsaPackQkv`** — repacks a QKV conv's NCHW output (`[B, qkv_h, H, W]`,
  channels laid out per-head as `[Q | K | V_lo | V_hi]`, each `KEY_DIM`
  wide) into the `[4, BH, N, KEY_DIM]` layout Flash Attention 2 expects.
- **`PsaExtractV`** — pulls the V section back out in NCHW for the residual
  path.
- **`PsaMergeAttn`** — merges attention output back into NCHW `[B, c, H,
  W]`.

The `V_lo`/`V_hi` split exists because `head_dim = 2 * key_dim` in the
ultralytics PSABlock; rather than run Flash Attention 2 once with
`HEAD_DIM = head_dim`, PSA runs it *twice* with `HEAD_DIM = key_dim` (once
per half), avoiding a HEAD_DIM value that isn't a clean power-of-two
multiple of the underlying key dimension in all configurations.

Each of `PsaPackQkvOp`/`PsaExtractVOp`/`PsaMergeAttnOp`/`FlashAttn2PsaOp` is
a thin `CustomOp` wrapper (see
[Building Models](https://docs.teenygrad.org/teenygrad/nn-layers/building-models)
in the teenygrad book) that records a graph node directly — no separate
lowering middleware needed, since `lower()` just hands the pre-built
`Arc<RuntimeOp>` straight through.

## Detect-decode

`kernels::detect_decode` converts a model's raw LTRB box predictions plus a
precomputed anchor grid into decoded `[cx, cy, w, h]` boxes in one fused
kernel pass, rather than doing the anchor-grid arithmetic on the host.
`DetectDecodeOp` carries the anchor grid (`anchor_x`, `anchor_y`,
`strides`) as graph-node state; `DetectDecodeRuntimeOp` uploads that data to
device parameter buffers at model-load time via
`RuntimeOp::param_init_data`, so it's a one-time setup cost, not a
per-inference one.

## Loss kernels

`kernels::loss` has the CUDA forward/backward kernels
[`Yolo26Loss`](../core-concepts/training.md) dispatches:

- **`ciou`** — fused CIoU loss: takes predicted/target `[4, N]` XYWH boxes,
  produces per-anchor loss plus the intermediate `iou`/`v`/`alpha` values
  the backward pass needs (saved-activation pattern, avoiding
  recomputation). One CTA per `BLOCK_N`-wide anchor tile.
- **`cls`** — the classification loss (BCE-based) forward/backward kernels.

## Compiling and inspecting kernel source

Every `#[kernel]`-annotated function generates a struct (e.g.
`FlashAttention2Forward<D>`) implementing teenygrad's `Kernel` trait, with
`.source()`/`.name` giving you the generated Rust source and entry-point
name — useful when debugging a kernel change, or when writing a snapshot
test against the generated MLIR/source (see the `test_*` files under
`tests/` for examples of both).
