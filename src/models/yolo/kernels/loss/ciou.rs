//! Fused CIoU loss Triton kernel for YOLO detection training.
//!
//! Computes Complete-IoU (CIoU) loss between predicted and target boxes,
//! both in XYWH world-coordinate format, in a single fused kernel pass.
//!
//! CIoU = 1 − IoU + d²/c² + α·v
//!
//! where:
//!   - IoU  = intersection / union
//!   - d²   = squared Euclidean distance between box centers
//!   - c²   = squared diagonal of the smallest enclosing box
//!   - v    = (4/π²) · (atan(tw/th) − atan(pw/ph))²  (aspect-ratio consistency)
//!   - α    = v / (1 − IoU + v + ε)
//!
//! Layout:
//!   - `pred`:   `[4, N]` — predicted (cx, cy, w, h) per anchor
//!   - `target`: `[4, N]` — target (cx, cy, w, h) per anchor
//!   - `loss`:   `[N]`    — per-anchor CIoU loss
//!
//! Parallelism: one CTA per BLOCK_N-wide anchor tile.
//! Grid: `cdiv(N, BLOCK_N)` flat CTAs.

#![allow(non_snake_case, clippy::erasing_op, clippy::identity_op)]

use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

/// Fused CIoU loss forward: predicted and target XYWH boxes → per-anchor loss.
///
/// Grid: `cdiv(N, BLOCK_N)` — one CTA per anchor tile.
#[kernel]
pub fn yolo_ciou_loss_forward<T: Triton, const BLOCK_N: i32>(
    pred_ptr: T::Pointer<f32>,
    target_ptr: T::Pointer<f32>,
    loss_ptr: T::Pointer<f32>,
    N: i32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let n_start = T::program_id(Axis::X) * BLOCK_N;
    let n_offs  = T::arange(0, BLOCK_N) + n_start;
    let mask    = n_offs.lt(N);
    let zeros   = T::zeros::<f32>(&[BLOCK_N]);

    // Pred XYWH — layout [4, N]: channel c is at base offset c*N.
    let px = T::load(pred_ptr.add_offsets(n_offs + 0 * N), Some(mask), Some(zeros), &[], None, None, None, false);
    let py = T::load(pred_ptr.add_offsets(n_offs + 1 * N), Some(mask), Some(zeros), &[], None, None, None, false);
    let pw = T::load(pred_ptr.add_offsets(n_offs + 2 * N), Some(mask), Some(zeros), &[], None, None, None, false);
    let ph = T::load(pred_ptr.add_offsets(n_offs + 3 * N), Some(mask), Some(zeros), &[], None, None, None, false);

    // Target XYWH.
    let tx = T::load(target_ptr.add_offsets(n_offs + 0 * N), Some(mask), Some(zeros), &[], None, None, None, false);
    let ty = T::load(target_ptr.add_offsets(n_offs + 1 * N), Some(mask), Some(zeros), &[], None, None, None, false);
    let tw = T::load(target_ptr.add_offsets(n_offs + 2 * N), Some(mask), Some(zeros), &[], None, None, None, false);
    let th = T::load(target_ptr.add_offsets(n_offs + 3 * N), Some(mask), Some(zeros), &[], None, None, None, false);

    let half = T::full::<f32>(&[BLOCK_N], 0.5f32);
    let eps  = T::full::<f32>(&[BLOCK_N], 1e-7f32);
    let ones = T::full::<f32>(&[BLOCK_N], 1.0f32);

    // Pred corners.
    let px1 = px - pw * half;
    let px2 = px + pw * half;
    let py1 = py - ph * half;
    let py2 = py + ph * half;

    // Target corners.
    let tx1 = tx - tw * half;
    let tx2 = tx + tw * half;
    let ty1 = ty - th * half;
    let ty2 = ty + th * half;

    // Intersection.
    let ix1 = T::maximum(px1, tx1);
    let ix2 = T::minimum(px2, tx2);
    let iy1 = T::maximum(py1, ty1);
    let iy2 = T::minimum(py2, ty2);
    let inter_w = T::maximum(ix2 - ix1, zeros);
    let inter_h = T::maximum(iy2 - iy1, zeros);
    let inter   = inter_w * inter_h;

    // Union.
    let pred_area   = pw * ph;
    let target_area = tw * th;
    let union = pred_area + target_area - inter;

    // IoU.
    let iou = inter / (union + eps);

    // Center distance squared.
    let dx = px - tx;
    let dy = py - ty;
    let d2 = dx * dx + dy * dy;

    // Smallest enclosing box diagonal squared.
    let ex1 = T::minimum(px1, tx1);
    let ex2 = T::maximum(px2, tx2);
    let ey1 = T::minimum(py1, ty1);
    let ey2 = T::maximum(py2, ty2);
    let ecw = ex2 - ex1;
    let ech = ey2 - ey1;
    let c2  = ecw * ecw + ech * ech;

    // Aspect-ratio consistency term (requires atan).
    let atan_t = T::atan(tw / (th + eps));
    let atan_p = T::atan(pw / (ph + eps));
    let diff   = atan_t - atan_p;
    // 4 / π²  ≈ 0.405284734
    let pi2_inv4 = T::full::<f32>(&[BLOCK_N], 0.405_284_73_f32);
    let v = pi2_inv4 * diff * diff;

    // α = v / (1 − IoU + v + ε)
    let alpha = v / (ones - iou + v + eps);

    // CIoU loss = 1 − IoU + d²/(c² + ε) + α·v
    let loss = ones - iou + d2 / (c2 + eps) + alpha * v;

    T::store(loss_ptr.add_offsets(n_offs), loss, Some(mask), &[], None, None);
}
