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


//! NCHW ↔ NLD reshape kernels for DINOv2 patch embedding.
//!
//! `dinov2_nchw_to_nld`: `[B, D, H, W]` → `[B, N, D]` where `N = H × W`.
//!   Used to convert patch embeddings to sequence format for the ViT.
//!
//! `dinov2_nld_to_nchw`: `[B, N, D]` → `[B, D, H, W]`.
//!   Used to convert ViT feature tokens back to spatial format for the decoder.
//!
//! Each CTA covers one `(b, n)` pair; the `D` elements are loaded/stored
//! sequentially (NLD side) or with stride `n_spatial` (NCHW side).

#![allow(non_snake_case)]

use std::sync::Arc;

use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

// ── dinov2_nchw_to_nld ───────────────────────────────────────────────────────

/// Permutes `[B, D, H, W]` (NCHW) to `[B, N, D]` (NLD) where `N = H × W`.
///
/// Each CTA processes one `(b, n)` pair and reads/writes `D` elements:
/// - **Source** (strided):  `in[b, d, h, w]` = offset `b·D·N + d·N + n`
/// - **Dest** (sequential): `out[b, n, d]`   = offset `(b·N + n)·D + d`
///
/// Grid: `[B × N, 1, 1]`.  Block: Triton-emitted 128 threads (4 warps).
///
/// `BLOCK_D` must be a power of 2; `embed_dim` is the actual dimension (≤ BLOCK_D).
#[kernel]
pub fn dinov2_nchw_to_nld<T: Triton, const BLOCK_D: i32>(
    in_ptr:    T::Pointer<f32>,  // [B, D, H, W]
    out_ptr:   T::Pointer<f32>,  // [B, N, D]
    n_spatial: i32,              // N = H * W
    embed_dim: i32,              // actual D (≤ BLOCK_D)
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid  = T::program_id(Axis::X); // [0, B * N)
    let d    = T::arange(0, BLOCK_D);
    let mask = d.lt(embed_dim);

    let b = pid / n_spatial;
    let n = pid - b * n_spatial;

    // Source: in[b, d, h, w] — NCHW, stride n_spatial between D elements.
    // in_offset[d] = b * embed_dim * n_spatial + d * n_spatial + n
    let in_base = b * embed_dim * n_spatial + n;
    let zeros   = T::zeros::<f32>(&[BLOCK_D]);
    let x = T::load(in_ptr.add_offsets(d * n_spatial + in_base), Some(mask), Some(zeros), &[], None, None, None, false);

    // Destination: out[b, n, d] — NLD, sequential.
    let out_base = (b * n_spatial + n) * embed_dim;
    T::store(out_ptr.add_offsets(d + out_base), x, Some(mask), &[], None, None);
}

// ── dinov2_nld_to_nchw ───────────────────────────────────────────────────────

/// Permutes `[B, N, D]` (NLD) to `[B, D, H, W]` (NCHW) where `N = H × W`.
///
/// The transpose of `dinov2_nchw_to_nld`.  Used to reshape ViT feature tokens
/// into spatial feature maps for the MultiScaleProjector.
///
/// Grid: `[B × N, 1, 1]`.  Block: Triton-emitted 128 threads (4 warps).
///
/// `BLOCK_D` must be a power of 2; `embed_dim` is the actual dimension (≤ BLOCK_D).
#[kernel]
pub fn dinov2_nld_to_nchw<T: Triton, const BLOCK_D: i32>(
    in_ptr:    T::Pointer<f32>,  // [B, N, D]
    out_ptr:   T::Pointer<f32>,  // [B, D, H, W]
    n_spatial: i32,              // N = H * W
    embed_dim: i32,              // actual D (≤ BLOCK_D)
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid  = T::program_id(Axis::X); // [0, B * N)
    let d    = T::arange(0, BLOCK_D);
    let mask = d.lt(embed_dim);

    let b = pid / n_spatial;
    let n = pid - b * n_spatial;

    // Source: in[b, n, d] — NLD, sequential.
    let in_base = (b * n_spatial + n) * embed_dim;
    let zeros   = T::zeros::<f32>(&[BLOCK_D]);
    let x = T::load(in_ptr.add_offsets(d + in_base), Some(mask), Some(zeros), &[], None, None, None, false);

    // Destination: out[b, d, h, w] — NCHW, stride n_spatial.
    // out_offset[d] = b * embed_dim * n_spatial + d * n_spatial + n
    let out_base = b * embed_dim * n_spatial + n;
    T::store(out_ptr.add_offsets(d * n_spatial + out_base), x, Some(mask), &[], None, None);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn next_pow2(n: usize) -> usize {
    if n == 0 { return 1; }
    let mut p = 1usize;
    while p < n { p <<= 1; }
    p
}

// ── RuntimeOp: Dinov2NchwToNldRuntimeOp ──────────────────────────────────────

pub struct Dinov2NchwToNldRuntimeOp {
    fwd:       Dinov2NchwToNld,
    bwd:       Dinov2NldToNchw,
    embed_dim: usize,
}

impl Dinov2NchwToNldRuntimeOp {
    pub fn new(embed_dim: i32) -> Self {
        let block_d = next_pow2(embed_dim as usize) as i32;
        Self {
            fwd:       Dinov2NchwToNld::new(block_d),
            bwd:       Dinov2NldToNchw::new(block_d),
            embed_dim: embed_dim as usize,
        }
    }

    pub fn kernel_name(&self)     -> &str { self.fwd.name }
    pub fn forward_source(&self)  -> &str { &self.fwd.source }
    pub fn backward_source(&self) -> &str { &self.bwd.source }
}

impl teeny_core::model::RuntimeOp for Dinov2NchwToNldRuntimeOp {
    fn n_activation_inputs(&self) -> usize { 1 }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { vec![] }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // inputs[0].1 = [B, D, H, W]
        let h = inputs[0].1[2];
        let w = inputs[0].1[3];
        let n_spatial = (h * w) as i32;

        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n_spatial);
        visitor.visit_i32(self.embed_dim as i32);
    }

    fn block(&self) -> [u32; 3] { [1, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [B, N, D]; grid = (B * N, 1, 1)
        [(output_shape[0] * output_shape[1]) as u32, 1, 1]
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
        let h = inputs[0].1[2];
        let w = inputs[0].1[3];
        let n_spatial = (h * w) as i32;

        // bwd kernel = nld_to_nchw: d_packed_NLD → d_input_NCHW
        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_i32(n_spatial);
        visitor.visit_i32(self.embed_dim as i32);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [1, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, input_shapes: &[&[usize]], _: &[usize]) -> [u32; 3] {
        // grad goes back to [B, D, H, W]; grid = (B * H * W, 1, 1)
        let h = input_shapes[0][2];
        let w = input_shapes[0][3];
        [(input_shapes[0][0] * h * w) as u32, 1, 1]
    }
}

// ── RuntimeOp: Dinov2NldToNchwRuntimeOp ──────────────────────────────────────

pub struct Dinov2NldToNchwRuntimeOp {
    fwd:       Dinov2NldToNchw,
    bwd:       Dinov2NchwToNld,
    embed_dim: usize,
}

impl Dinov2NldToNchwRuntimeOp {
    pub fn new(embed_dim: i32) -> Self {
        let block_d = next_pow2(embed_dim as usize) as i32;
        Self {
            fwd:       Dinov2NldToNchw::new(block_d),
            bwd:       Dinov2NchwToNld::new(block_d),
            embed_dim: embed_dim as usize,
        }
    }

    pub fn kernel_name(&self)     -> &str { self.fwd.name }
    pub fn forward_source(&self)  -> &str { &self.fwd.source }
    pub fn backward_source(&self) -> &str { &self.bwd.source }
}

impl teeny_core::model::RuntimeOp for Dinov2NldToNchwRuntimeOp {
    fn n_activation_inputs(&self) -> usize { 1 }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { vec![] }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // inputs[0].1 = [B, N, D]
        let n_spatial = inputs[0].1[1] as i32;

        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n_spatial);
        visitor.visit_i32(self.embed_dim as i32);
    }

    fn block(&self) -> [u32; 3] { [1, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [B, D, H, W]; grid = (B * H * W, 1, 1)
        [(output_shape[0] * output_shape[2] * output_shape[3]) as u32, 1, 1]
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
        let n_spatial = inputs[0].1[1] as i32;

        // bwd kernel = nchw_to_nld: d_NCHW → d_input_NLD
        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_i32(n_spatial);
        visitor.visit_i32(self.embed_dim as i32);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [1, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, input_shapes: &[&[usize]], _: &[usize]) -> [u32; 3] {
        // grad goes back to [B, N, D]; grid = (B * N, 1, 1)
        [(input_shapes[0][0] * input_shapes[0][1]) as u32, 1, 1]
    }
}

// ── CustomOp: Dinov2NchwToNldOp ───────────────────────────────────────────────

use std::any::Any;
use teeny_core::{
    graph::{CustomOp, Shape},
    model::RuntimeOp,
};

/// Graph node: `[B, D, H, W]` → `[B, H×W, D]`.
pub struct Dinov2NchwToNldOp {
    inner:     Arc<Dinov2NchwToNldRuntimeOp>,
    embed_dim: usize,
}

impl Dinov2NchwToNldOp {
    pub fn new(embed_dim: i32) -> Self {
        Self {
            inner:     Arc::new(Dinov2NchwToNldRuntimeOp::new(embed_dim)),
            embed_dim: embed_dim as usize,
        }
    }
}

impl CustomOp for Dinov2NchwToNldOp {
    fn name(&self) -> &str { "dinov2_nchw_to_nld" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // input: [B, D, H, W]  → output: [B, N, D]
        let b = input_shapes[0][0];
        let h = input_shapes[0][2];
        let w = input_shapes[0][3];
        let n = h.zip(w).map(|(hv, wv)| hv * wv);
        vec![b, n, Some(self.embed_dim)]
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

// ── CustomOp: Dinov2NldToNchwOp ───────────────────────────────────────────────

/// Graph node: `[B, N, D]` → `[B, D, H, W]`.  Caller must supply target H, W.
pub struct Dinov2NldToNchwOp {
    inner:     Arc<Dinov2NldToNchwRuntimeOp>,
    embed_dim: usize,
    h_patches: usize,
    w_patches: usize,
}

impl Dinov2NldToNchwOp {
    pub fn new(embed_dim: i32, h_patches: usize, w_patches: usize) -> Self {
        Self {
            inner:     Arc::new(Dinov2NldToNchwRuntimeOp::new(embed_dim)),
            embed_dim: embed_dim as usize,
            h_patches,
            w_patches,
        }
    }
}

impl CustomOp for Dinov2NldToNchwOp {
    fn name(&self) -> &str { "dinov2_nld_to_nchw" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // input: [B, N, D] → output: [B, D, H, W]
        let b = input_shapes[0][0];
        vec![b, Some(self.embed_dim), Some(self.h_patches), Some(self.w_patches)]
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
