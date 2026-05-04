//! BCE classification loss Triton kernel for YOLO detection training.
//!
//! Computes numerically-stable binary cross-entropy loss between predicted
//! class logits and soft target labels, summed over all C classes per anchor.
//!
//! Stable BCE formula (avoids log(0) and exp overflow):
//!   loss = relu(x) − x·t + log(1 + exp(−|x|))
//!        = max(x, 0) − x·t + log(1 + exp(−|x|))
//!
//! Layout:
//!   - `pred`:   `[C, N]` — predicted class logits per anchor
//!   - `target`: `[C, N]` — soft class labels ∈ [0, 1] per anchor
//!   - `loss`:   `[N]`    — per-anchor BCE loss (summed over C classes)
//!
//! Parallelism: one CTA per BLOCK_N-wide anchor tile; classes iterated
//! sequentially inside each CTA.
//! Grid: `cdiv(N, BLOCK_N)` flat CTAs.

#![allow(non_snake_case)]

use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

/// BCE classification loss forward: class logits + soft targets → per-anchor loss.
///
/// Grid: `cdiv(N, BLOCK_N)` — one CTA per anchor tile.
#[allow(clippy::erasing_op, clippy::identity_op)]
#[kernel]
pub fn yolo_bce_cls_loss_forward<T: Triton, const BLOCK_N: i32>(
    pred_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    loss_ptr: T::Pointer<f32>,
    N: i32,
    C: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let n_start = T::program_id(Axis::X) * BLOCK_N;
    let n_offs  = T::arange(0, BLOCK_N) + n_start;
    let mask    = n_offs.lt(N);
    let zeros   = T::zeros::<f32>(&[BLOCK_N]);
    let ones    = T::full::<f32>(&[BLOCK_N], 1.0f32);

    // Accumulate BCE over all C classes for this anchor tile.
    let mut acc = zeros;
    let mut c: i32 = 0;
    while c < C {
        // Layout [C, N]: class c occupies row c, at base offset c*N.
        let base = c * N;
        let x = T::load(pred_ptr.add_offsets(n_offs + base),   Some(mask), Some(zeros), &[], None, None, None, false);
        let t = T::load(target_ptr.add_offsets(n_offs + base), Some(mask), Some(zeros), &[], None, None, None, false);

        // Numerically-stable BCE: max(x, 0) − x·t + log(1 + exp(−|x|))
        let relu_x    = T::maximum(x, zeros);
        let log1p_exp = T::log(ones + T::exp(zeros - T::abs(x)));
        let bce       = relu_x - x * t + log1p_exp;

        acc = acc + bce;
        c += 1;
    }

    T::store(loss_ptr.add_offsets(n_offs), acc, Some(mask), &[], None, None);
}
