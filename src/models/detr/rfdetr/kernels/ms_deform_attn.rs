/*
 * Copyright 2026 Teenygrad
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


//! Multi-Scale Deformable Attention (MSDeformAttn) Triton kernels.
//!
//! Implements the bilinear grid sampling + weighted sum used by RF-DETR's
//! cross-attention decoder layers.  One CTA handles one (query, batch-head)
//! pair; the channel dimension `HEAD_DIM` is processed as a 1-D tile.
//!
//! Layouts (all row-major):
//!   - `value`:          `[BH, S_total, HEAD_DIM]`
//!   - `sampling_locs`:  `[BH, Nq, n_levels, n_points, 2]`  (lx, ly in [0, 1])
//!   - `attn_weights`:   `[BH, Nq, n_levels * n_points]`    (post-softmax)
//!   - `spatial_shapes`: `[n_levels, 2]`                    (Hl, Wl as f32)
//!   - `level_start`:    `[n_levels]`                       (cumsum Hl*Wl as f32)
//!   - `output`:         `[BH, Nq, HEAD_DIM]`
//!
//! Pixel coordinate convention (matches PyTorch grid_sample align_corners=False
//! composed with the `2*lx-1` normalisation used in rf-detr):
//!   `px = lx * (Wl - 1)`,  `py = ly * (Hl - 1)`
//! Pixels outside `[0, Hl) × [0, Wl)` contribute 0 (zero-padding boundary).

#![allow(non_snake_case)]

use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

// ── ms_deform_attn_forward ────────────────────────────────────────────────────

/// Forward kernel: bilinear-sample value at each (level, point) location and
/// accumulate the attention-weighted sum for each query.
///
/// Grid:  `(Nq, BH, 1)` — one CTA per (query, batch-head).
/// Block: `(HEAD_DIM, 1, 1)` — one thread per channel (set in RuntimeOp).
#[kernel]
pub fn ms_deform_attn_forward<T: Triton, const HEAD_DIM: i32>(
    value_ptr: T::Pointer<f32>,          // [BH, S_total, HEAD_DIM]
    sampling_locs_ptr: T::Pointer<f32>,  // [BH, Nq, n_levels, n_points, 2]
    attn_weights_ptr: T::Pointer<f32>,   // [BH, Nq, n_levels * n_points]
    spatial_shapes_ptr: T::Pointer<f32>, // [n_levels, 2] (Hl, Wl) stored as f32
    level_start_ptr: T::Pointer<f32>,    // [n_levels] cumulative starts as f32
    output_ptr: T::Pointer<f32>,         // [BH, Nq, HEAD_DIM]
    Nq: i32,
    S_total: i32,
    n_levels: i32,
    n_points: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
    T::Tensor<i32>: Into<T::I32Tensor>,
    T::I32Tensor: Into<T::Tensor<i32>>,
{
    let pid_q  = T::program_id(Axis::X); // [0, Nq)
    let pid_bh = T::program_id(Axis::Y); // [0, BH)

    // Channel tile: [0, HEAD_DIM)
    let d = T::arange(0, HEAD_DIM);          // T::I32Tensor
    let d_t: T::Tensor<i32> = d.into();      // for arithmetic with T::Tensor<i32>

    let mut acc = T::zeros::<f32>(&[HEAD_DIM]);

    // Base offset into value buffer for this batch-head (scalar, avoid overflow)
    let bh_val_base: i32 = pid_bh * S_total * HEAD_DIM;

    // Flat loop over all (level, point) pairs to avoid nested-loop backend limitations.
    // l = lp / n_points,  p = lp - l * n_points
    let nlp = n_levels * n_points;
    for lp in 0..nlp {
        let l       = lp / n_points;
        let p       = lp - l * n_points;

        // Spatial shape and start for level l (integer values stored as f32)
        let hl      = T::load_scalar_f32_as_i32(spatial_shapes_ptr, l * 2);
        let wl      = T::load_scalar_f32_as_i32(spatial_shapes_ptr, l * 2 + 1);
        let start_l = T::load_scalar_f32_as_i32(level_start_ptr, l);

        let wl_t    = T::full::<i32>(&[1], wl);
        let hl_t    = T::full::<i32>(&[1], hl);
        let zero_i  = T::full::<i32>(&[1], 0);
        let wl_m1_i = T::full::<i32>(&[1], wl - 1);
        let hl_m1_i = T::full::<i32>(&[1], hl - 1);
        let hd_t    = T::full::<i32>(&[1], HEAD_DIM);
        let wl_m1_f = T::full::<f32>(&[1], (wl - 1) as f32);
        let hl_m1_f = T::full::<f32>(&[1], (hl - 1) as f32);
        let zero_f  = T::full::<f32>(&[1], 0.0_f32);
        let one_f   = T::full::<f32>(&[1], 1.0_f32);

        // Scalar base offset for this (bh, level) in the value buffer
        let bh_lvl_base: i32 = bh_val_base + start_l * HEAD_DIM;

        {
            // ── Load sampling location (lx, ly) ──────────────────────────────
            let loc_off: i32 = pid_bh * (Nq * n_levels * n_points * 2)
                             + pid_q  * (n_levels * n_points * 2)
                             + l      * (n_points * 2)
                             + p      * 2;

            // 1-element loads: use arange(0,1) + runtime_offset so that
            // arange's bounds are compile-time constants (MLIR backend requirement).
            let lx_t = T::load(
                sampling_locs_ptr.add_offsets(T::arange(0, 1) + loc_off),
                None, None, &[], None, None, None, false,
            ); // T::Tensor<f32>, shape [1]
            let ly_t = T::load(
                sampling_locs_ptr.add_offsets(T::arange(0, 1) + (loc_off + 1)),
                None, None, &[], None, None, None, false,
            );

            // ── Pixel coordinates ─────────────────────────────────────────────
            // px = lx * (Wl - 1),  py = ly * (Hl - 1)
            let px_t = lx_t * wl_m1_f; // T::Tensor<f32>, shape [1]
            let py_t = ly_t * hl_m1_f;

            // Floor to base-corner integer coords
            let x0_f = T::floor(px_t);
            let y0_f = T::floor(py_t);

            // Bilinear fractional weights
            let wx1_t = px_t - x0_f;
            let wy1_t = py_t - y0_f;
            let wx0_t = one_f - wx1_t;
            let wy0_t = one_f - wy1_t;

            // Cast to integer coords (T::Tensor<i32>, shape [1])
            let x0_i = T::cast::<f32, i32>(x0_f, None, false);
            let y0_i = T::cast::<f32, i32>(y0_f, None, false);
            let x1_i = x0_i + T::full::<i32>(&[1], 1);
            let y1_i = y0_i + T::full::<i32>(&[1], 1);

            // ── Boundary masks (zero OOB contributions) ───────────────────────
            let mask_x0 = T::ge(x0_i, zero_i) & T::lt(x0_i, wl_t);
            let mask_x1 = T::ge(x1_i, zero_i) & T::lt(x1_i, wl_t);
            let mask_y0 = T::ge(y0_i, zero_i) & T::lt(y0_i, hl_t);
            let mask_y1 = T::ge(y1_i, zero_i) & T::lt(y1_i, hl_t);

            // Masked bilinear weights (0 for OOB corners)
            let wx0_m = T::where_(mask_x0, wx0_t, zero_f);
            let wx1_m = T::where_(mask_x1, wx1_t, zero_f);
            let wy0_m = T::where_(mask_y0, wy0_t, zero_f);
            let wy1_m = T::where_(mask_y1, wy1_t, zero_f);

            // Clamp indices for safe memory access (OOB zeroed via weight anyway).
            // Both clamp and maximum emit float-only ops; use where+minimum instead.
            let x0_s = T::where_::<i32>(T::lt(x0_i, zero_i), zero_i, T::minimum(wl_m1_i, x0_i));
            let x1_s = T::where_::<i32>(T::lt(x1_i, zero_i), zero_i, T::minimum(wl_m1_i, x1_i));
            let y0_s = T::where_::<i32>(T::lt(y0_i, zero_i), zero_i, T::minimum(hl_m1_i, y0_i));
            let y1_s = T::where_::<i32>(T::lt(y1_i, zero_i), zero_i, T::minimum(hl_m1_i, y1_i));

            // ── Value offsets: bh_lvl_base + (y*Wl + x)*HEAD_DIM + d ─────────
            // Build the spatial part as a [1] tensor, then broadcast → [HEAD_DIM]
            // before adding d_t ([HEAD_DIM]).  This prevents the backend from
            // incorrectly emitting tt.broadcast on d_t itself.
            let off00: T::Tensor<i32> = d_t + T::broadcast_to(T::full::<i32>(&[1], bh_lvl_base) + (y0_s * wl_t + x0_s) * hd_t, &[HEAD_DIM]);
            let off01: T::Tensor<i32> = d_t + T::broadcast_to(T::full::<i32>(&[1], bh_lvl_base) + (y0_s * wl_t + x1_s) * hd_t, &[HEAD_DIM]);
            let off10: T::Tensor<i32> = d_t + T::broadcast_to(T::full::<i32>(&[1], bh_lvl_base) + (y1_s * wl_t + x0_s) * hd_t, &[HEAD_DIM]);
            let off11: T::Tensor<i32> = d_t + T::broadcast_to(T::full::<i32>(&[1], bh_lvl_base) + (y1_s * wl_t + x1_s) * hd_t, &[HEAD_DIM]);

            // Load value at 4 corners (safe: indices clamped; OOB zeroed by weight)
            let val00 = T::load(value_ptr.add_offsets(off00.into()), None, None, &[], None, None, None, false);
            let val01 = T::load(value_ptr.add_offsets(off01.into()), None, None, &[], None, None, None, false);
            let val10 = T::load(value_ptr.add_offsets(off10.into()), None, None, &[], None, None, None, false);
            let val11 = T::load(value_ptr.add_offsets(off11.into()), None, None, &[], None, None, None, false);

            // ── Bilinear interpolation ────────────────────────────────────────
            let bilinear = T::broadcast_to(wy0_m * wx0_m, &[HEAD_DIM]) * val00
                         + T::broadcast_to(wy0_m * wx1_m, &[HEAD_DIM]) * val01
                         + T::broadcast_to(wy1_m * wx0_m, &[HEAD_DIM]) * val10
                         + T::broadcast_to(wy1_m * wx1_m, &[HEAD_DIM]) * val11;

            // ── Attention weight for this (level, point) ──────────────────────
            let w_off: i32 = pid_bh * (Nq * n_levels * n_points)
                           + pid_q  * (n_levels * n_points)
                           + l * n_points + p;
            let w_t = T::load(
                attn_weights_ptr.add_offsets(T::arange(0, 1) + w_off),
                None, None, &[], None, None, None, false,
            ); // T::Tensor<f32>, shape [1]

            acc = acc + T::broadcast_to(w_t, &[HEAD_DIM]) * bilinear;
        }
    }

    // ── Write output ──────────────────────────────────────────────────────────
    let out_base: i32 = pid_bh * Nq * HEAD_DIM + pid_q * HEAD_DIM;
    T::store(output_ptr.add_offsets(d + out_base), acc, None, &[], None, None);
}

// ── ms_deform_attn_backward ───────────────────────────────────────────────────

/// Backward kernel: computes gradients for value, sampling_locs, and attn_weights.
///
/// Grid:  `(Nq, BH, 1)` — same as forward.
/// Block: `(HEAD_DIM, 1, 1)`.
///
/// `d_value` is updated with `atomic_add` since multiple (q, l, p) can map to
/// the same spatial position.  `d_sampling_locs` and `d_attn_weights` are
/// written with regular stores (each (q, l, p) owns a unique output location).
#[kernel]
pub fn ms_deform_attn_backward<T: Triton, const HEAD_DIM: i32>(
    value_ptr: T::Pointer<f32>,           // [BH, S_total, HEAD_DIM] — saved fwd input
    sampling_locs_ptr: T::Pointer<f32>,   // [BH, Nq, n_levels, n_points, 2]
    attn_weights_ptr: T::Pointer<f32>,    // [BH, Nq, n_levels * n_points]
    spatial_shapes_ptr: T::Pointer<f32>,  // [n_levels, 2]
    level_start_ptr: T::Pointer<f32>,     // [n_levels]
    grad_output_ptr: T::Pointer<f32>,     // [BH, Nq, HEAD_DIM] — upstream gradient
    d_value_ptr: T::Pointer<f32>,         // [BH, S_total, HEAD_DIM] — atomic_add target
    d_sampling_locs_ptr: T::Pointer<f32>, // [BH, Nq, n_levels, n_points, 2]
    d_attn_weights_ptr: T::Pointer<f32>,  // [BH, Nq, n_levels * n_points]
    Nq: i32,
    S_total: i32,
    n_levels: i32,
    n_points: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
    T::Tensor<i32>: Into<T::I32Tensor>,
    T::I32Tensor: Into<T::Tensor<i32>>,
{
    let pid_q  = T::program_id(Axis::X);
    let pid_bh = T::program_id(Axis::Y);

    let d = T::arange(0, HEAD_DIM);         // T::I32Tensor
    let d_t: T::Tensor<i32> = d.into();

    // Load upstream gradient for this (bh, q)
    let go_base: i32 = pid_bh * Nq * HEAD_DIM + pid_q * HEAD_DIM;
    let d_out = T::load(grad_output_ptr.add_offsets(d + go_base), None, None, &[], None, None, None, false);
    // d_out: T::Tensor<f32>, shape [HEAD_DIM]

    let bh_val_base: i32 = pid_bh * S_total * HEAD_DIM;

    // Flat loop over all (level, point) pairs to avoid nested-loop backend limitations.
    // l = lp / n_points,  p = lp - l * n_points
    let nlp = n_levels * n_points;
    for lp in 0..nlp {
        let l       = lp / n_points;
        let p       = lp - l * n_points;

        let hl      = T::load_scalar_f32_as_i32(spatial_shapes_ptr, l * 2);
        let wl      = T::load_scalar_f32_as_i32(spatial_shapes_ptr, l * 2 + 1);
        let start_l = T::load_scalar_f32_as_i32(level_start_ptr, l);

        let wl_t    = T::full::<i32>(&[1], wl);
        let hl_t    = T::full::<i32>(&[1], hl);
        let zero_i  = T::full::<i32>(&[1], 0);
        let wl_m1_i = T::full::<i32>(&[1], wl - 1);
        let hl_m1_i = T::full::<i32>(&[1], hl - 1);
        let hd_t    = T::full::<i32>(&[1], HEAD_DIM);
        let wl_m1_f = T::full::<f32>(&[1], (wl - 1) as f32);
        let hl_m1_f = T::full::<f32>(&[1], (hl - 1) as f32);
        let zero_f  = T::full::<f32>(&[1], 0.0_f32);
        let one_f   = T::full::<f32>(&[1], 1.0_f32);
        let bh_lvl_base: i32 = bh_val_base + start_l * HEAD_DIM;

        {
            let loc_off: i32 = pid_bh * (Nq * n_levels * n_points * 2)
                             + pid_q  * (n_levels * n_points * 2)
                             + l      * (n_points * 2)
                             + p      * 2;

            // ── Re-compute bilinear data (same as forward) ────────────────────
            let lx_t = T::load(
                sampling_locs_ptr.add_offsets(T::arange(0, 1) + loc_off),
                None, None, &[], None, None, None, false,
            );
            let ly_t = T::load(
                sampling_locs_ptr.add_offsets(T::arange(0, 1) + (loc_off + 1)),
                None, None, &[], None, None, None, false,
            );

            let px_t = lx_t * wl_m1_f;
            let py_t = ly_t * hl_m1_f;

            let x0_f = T::floor(px_t);
            let y0_f = T::floor(py_t);

            let wx1_t = px_t - x0_f;
            let wy1_t = py_t - y0_f;
            let wx0_t = one_f - wx1_t;
            let wy0_t = one_f - wy1_t;

            let x0_i = T::cast::<f32, i32>(x0_f, None, false);
            let y0_i = T::cast::<f32, i32>(y0_f, None, false);
            let x1_i = x0_i + T::full::<i32>(&[1], 1);
            let y1_i = y0_i + T::full::<i32>(&[1], 1);

            let mask_x0 = T::ge(x0_i, zero_i) & T::lt(x0_i, wl_t);
            let mask_x1 = T::ge(x1_i, zero_i) & T::lt(x1_i, wl_t);
            let mask_y0 = T::ge(y0_i, zero_i) & T::lt(y0_i, hl_t);
            let mask_y1 = T::ge(y1_i, zero_i) & T::lt(y1_i, hl_t);

            let wx0_m = T::where_(mask_x0, wx0_t, zero_f);
            let wx1_m = T::where_(mask_x1, wx1_t, zero_f);
            let wy0_m = T::where_(mask_y0, wy0_t, zero_f);
            let wy1_m = T::where_(mask_y1, wy1_t, zero_f);

            // Float existence masks (1.0 in-bounds, 0.0 OOB) for coord gradients
            let mf_x0 = T::where_(mask_x0, one_f, zero_f);
            let mf_x1 = T::where_(mask_x1, one_f, zero_f);
            let mf_y0 = T::where_(mask_y0, one_f, zero_f);
            let mf_y1 = T::where_(mask_y1, one_f, zero_f);

            let x0_s = T::where_::<i32>(T::lt(x0_i, zero_i), zero_i, T::minimum(wl_m1_i, x0_i));
            let x1_s = T::where_::<i32>(T::lt(x1_i, zero_i), zero_i, T::minimum(wl_m1_i, x1_i));
            let y0_s = T::where_::<i32>(T::lt(y0_i, zero_i), zero_i, T::minimum(hl_m1_i, y0_i));
            let y1_s = T::where_::<i32>(T::lt(y1_i, zero_i), zero_i, T::minimum(hl_m1_i, y1_i));

            let off00: T::Tensor<i32> = d_t + T::broadcast_to(T::full::<i32>(&[1], bh_lvl_base) + (y0_s * wl_t + x0_s) * hd_t, &[HEAD_DIM]);
            let off01: T::Tensor<i32> = d_t + T::broadcast_to(T::full::<i32>(&[1], bh_lvl_base) + (y0_s * wl_t + x1_s) * hd_t, &[HEAD_DIM]);
            let off10: T::Tensor<i32> = d_t + T::broadcast_to(T::full::<i32>(&[1], bh_lvl_base) + (y1_s * wl_t + x0_s) * hd_t, &[HEAD_DIM]);
            let off11: T::Tensor<i32> = d_t + T::broadcast_to(T::full::<i32>(&[1], bh_lvl_base) + (y1_s * wl_t + x1_s) * hd_t, &[HEAD_DIM]);

            let val00 = T::load(value_ptr.add_offsets(off00.into()), None, None, &[], None, None, None, false);
            let val01 = T::load(value_ptr.add_offsets(off01.into()), None, None, &[], None, None, None, false);
            let val10 = T::load(value_ptr.add_offsets(off10.into()), None, None, &[], None, None, None, false);
            let val11 = T::load(value_ptr.add_offsets(off11.into()), None, None, &[], None, None, None, false);

            // ── Attention weight ──────────────────────────────────────────────
            let w_off: i32 = pid_bh * (Nq * n_levels * n_points)
                           + pid_q  * (n_levels * n_points)
                           + l * n_points + p;
            let w_t = T::load(
                attn_weights_ptr.add_offsets(T::arange(0, 1) + w_off),
                None, None, &[], None, None, None, false,
            );

            // ── Gradient: d_attn_weights ──────────────────────────────────────
            // d_w = dot(d_out, bilinear_val) over HEAD_DIM
            let bilinear = T::broadcast_to(wy0_m * wx0_m, &[HEAD_DIM]) * val00
                         + T::broadcast_to(wy0_m * wx1_m, &[HEAD_DIM]) * val01
                         + T::broadcast_to(wy1_m * wx0_m, &[HEAD_DIM]) * val10
                         + T::broadcast_to(wy1_m * wx1_m, &[HEAD_DIM]) * val11;

            // keepdim=true so the result is tensor<1xf32> (store-compatible).
            // keepdim=false gives a scalar f32 which can't be passed to T::reshape.
            let d_w_1 = T::sum(d_out * bilinear, Some(0), true);
            T::store(
                d_attn_weights_ptr.add_offsets(T::arange(0, 1) + w_off),
                d_w_1, None, &[], None, None,
            );

            // ── Gradient: d_value (atomic scatter to 4 corners) ──────────────
            let d_contrib_scale = T::broadcast_to(w_t, &[HEAD_DIM]) * d_out;
            T::atomic_add(d_value_ptr.add_offsets(off00.into()),
                T::broadcast_to(wy0_m * wx0_m, &[HEAD_DIM]) * d_contrib_scale,
                None, None, None);
            T::atomic_add(d_value_ptr.add_offsets(off01.into()),
                T::broadcast_to(wy0_m * wx1_m, &[HEAD_DIM]) * d_contrib_scale,
                None, None, None);
            T::atomic_add(d_value_ptr.add_offsets(off10.into()),
                T::broadcast_to(wy1_m * wx0_m, &[HEAD_DIM]) * d_contrib_scale,
                None, None, None);
            T::atomic_add(d_value_ptr.add_offsets(off11.into()),
                T::broadcast_to(wy1_m * wx1_m, &[HEAD_DIM]) * d_contrib_scale,
                None, None, None);

            // ── Gradient: d_sampling_locs ─────────────────────────────────────
            // d(bilinear)/d(px): x-direction derivative of bilinear interp
            //   d/dpx = wy0*(d/dpx[wx1]*v01 + d/dpx[wx0]*v00)
            //         + wy1*(d/dpx[wx1]*v11 + d/dpx[wx0]*v10)
            //   where d/dpx[wx1]=1, d/dpx[wx0]=-1; masked by OOB indicator
            let d_bx_0 = T::sum(
                d_out * T::broadcast_to(wy0_m * mf_x1, &[HEAD_DIM]) * val01, Some(0), false,
            ) - T::sum(
                d_out * T::broadcast_to(wy0_m * mf_x0, &[HEAD_DIM]) * val00, Some(0), false,
            );
            let d_bx_1 = T::sum(
                d_out * T::broadcast_to(wy1_m * mf_x1, &[HEAD_DIM]) * val11, Some(0), false,
            ) - T::sum(
                d_out * T::broadcast_to(wy1_m * mf_x0, &[HEAD_DIM]) * val10, Some(0), false,
            );
            let d_px = w_t * (d_bx_0 + d_bx_1); // scalar tensor []

            // d(bilinear)/d(py): y-direction derivative
            let d_by_0 = T::sum(
                d_out * T::broadcast_to(wx0_m * mf_y1, &[HEAD_DIM]) * val10, Some(0), false,
            ) - T::sum(
                d_out * T::broadcast_to(wx0_m * mf_y0, &[HEAD_DIM]) * val00, Some(0), false,
            );
            let d_by_1 = T::sum(
                d_out * T::broadcast_to(wx1_m * mf_y1, &[HEAD_DIM]) * val11, Some(0), false,
            ) - T::sum(
                d_out * T::broadcast_to(wx1_m * mf_y0, &[HEAD_DIM]) * val01, Some(0), false,
            );
            let d_py = w_t * (d_by_0 + d_by_1);

            // Chain rule: lx -> px = lx * (Wl-1)
            let d_lx = T::reshape(d_px * wl_m1_f, &[1], false);
            let d_ly = T::reshape(d_py * hl_m1_f, &[1], false);

            T::store(
                d_sampling_locs_ptr.add_offsets(T::arange(0, 1) + loc_off),
                d_lx, None, &[], None, None,
            );
            T::store(
                d_sampling_locs_ptr.add_offsets(T::arange(0, 1) + (loc_off + 1)),
                d_ly, None, &[], None, None,
            );

        }
    }
}

// ── RuntimeOp ────────────────────────────────────────────────────────────────

pub struct MsDeformAttnRuntimeOp {
    fwd: MsDeformAttnForward,
    bwd: MsDeformAttnBackward,
    head_dim: usize,
    n_levels: usize,
    n_points: usize,
}

impl MsDeformAttnRuntimeOp {
    pub fn new(head_dim: i32, n_levels: usize, n_points: usize) -> Self {
        Self {
            fwd: MsDeformAttnForward::new(head_dim),
            bwd: MsDeformAttnBackward::new(head_dim),
            head_dim: head_dim as usize,
            n_levels,
            n_points,
        }
    }

    pub fn kernel_name(&self) -> &str { self.fwd.name }
    pub fn forward_source(&self) -> &str { &self.fwd.source }
    pub fn backward_source(&self) -> &str { &self.bwd.source }
}

impl teeny_core::model::RuntimeOp for MsDeformAttnRuntimeOp {
    // Differentiable inputs: value (0), sampling_locs (1), attn_weights (2)
    // Non-differentiable: spatial_shapes (3), level_start (4) — no gradient allocated
    fn n_activation_inputs(&self) -> usize { 3 }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { Vec::new() }

    fn compute_concrete_output_shape(&self, input_shapes: &[&[usize]], _resolved: &[usize]) -> Vec<usize> {
        // inputs[0] = value: [BH, S_total, HEAD_DIM]
        // inputs[1] = sampling_locs: [BH, Nq, ...]
        // output: [BH, Nq, HEAD_DIM]
        let bh = input_shapes[0][0];
        let nq = input_shapes[1][1];
        let hd = input_shapes[0][2];
        vec![bh, nq, hd]
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // inputs: [value, sampling_locs, attn_weights, spatial_shapes, level_start]
        // value:         [BH, S_total, HEAD_DIM]
        // sampling_locs: [BH, Nq, n_levels, n_points, 2]
        // attn_weights:  [BH, Nq, n_levels * n_points]
        let s = inputs[0].1;  // value shape
        let nq     = inputs[1].1[1] as i32;
        let s_total = s[1] as i32;

        visitor.visit_ptr(inputs[0].0);  // value_ptr
        visitor.visit_ptr(inputs[1].0);  // sampling_locs_ptr
        visitor.visit_ptr(inputs[2].0);  // attn_weights_ptr
        visitor.visit_ptr(inputs[3].0);  // spatial_shapes_ptr
        visitor.visit_ptr(inputs[4].0);  // level_start_ptr
        visitor.visit_ptr(output);       // output_ptr
        visitor.visit_i32(nq);
        visitor.visit_i32(s_total);
        visitor.visit_i32(self.n_levels as i32);
        visitor.visit_i32(self.n_points as i32);
    }

    fn block(&self) -> [u32; 3] { [self.head_dim as u32, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [BH, Nq, HEAD_DIM]
        // grid = (Nq, BH, 1)
        [output_shape[1] as u32, output_shape[0] as u32, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // grad_inputs: [d_value, d_sampling_locs, d_attn_weights]
        let s      = inputs[0].1;
        let nq     = inputs[1].1[1] as i32;
        let s_total = s[1] as i32;

        visitor.visit_ptr(inputs[0].0);      // value_ptr
        visitor.visit_ptr(inputs[1].0);      // sampling_locs_ptr
        visitor.visit_ptr(inputs[2].0);      // attn_weights_ptr
        visitor.visit_ptr(inputs[3].0);      // spatial_shapes_ptr
        visitor.visit_ptr(inputs[4].0);      // level_start_ptr
        visitor.visit_ptr(grad_output);      // grad_output_ptr
        visitor.visit_ptr(grad_inputs[0]);   // d_value_ptr
        visitor.visit_ptr(grad_inputs[1]);   // d_sampling_locs_ptr
        visitor.visit_ptr(grad_inputs[2]);   // d_attn_weights_ptr
        visitor.visit_i32(nq);
        visitor.visit_i32(s_total);
        visitor.visit_i32(self.n_levels as i32);
        visitor.visit_i32(self.n_points as i32);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [128, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [BH, Nq, HEAD_DIM]; grid = (Nq, BH, 1)
        [output_shape[1] as u32, output_shape[0] as u32, 1]
    }
}

// ── CustomOp ─────────────────────────────────────────────────────────────────

use std::any::Any;
use std::sync::Arc;
use teeny_core::{
    graph::{CustomOp, Shape},
    model::RuntimeOp,
};

/// Graph node: `[value, sampling_locs, attn_weights, spatial_shapes, level_start]`
/// → `[BH, Nq, HEAD_DIM]`
pub struct MsDeformAttnOp {
    inner: Arc<MsDeformAttnRuntimeOp>,
    head_dim: usize,
}

impl MsDeformAttnOp {
    pub fn new(head_dim: i32, n_levels: usize, n_points: usize) -> Self {
        Self {
            inner: Arc::new(MsDeformAttnRuntimeOp::new(head_dim, n_levels, n_points)),
            head_dim: head_dim as usize,
        }
    }
}

impl CustomOp for MsDeformAttnOp {
    fn name(&self) -> &str { "ms_deform_attn_forward" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // value: [BH, S_total, HEAD_DIM]; sampling_locs: [BH, Nq, ...]
        let bh = input_shapes[0][0];
        let nq = input_shapes[1][1];
        let hd = Some(self.head_dim);
        vec![bh, nq, hd]
    }

    fn as_any(&self) -> &dyn Any { self }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        Some((
            self.inner.kernel_name().to_string(),
            self.inner.forward_source().to_string(),
            "entry_point".to_string(),
            Arc::clone(&self.inner) as Arc<dyn RuntimeOp>,
        ))
    }

    fn lower_backward_source(&self) -> String {
        self.inner.backward_source().to_string()
    }
}
