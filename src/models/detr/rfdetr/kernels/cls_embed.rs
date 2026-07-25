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


//! Class-token and positional-embedding kernels for DINOv2.
//!
//! `dinov2_cat_cls`:       `[B, N, D]` + cls_token `[D]` → `[B, N+1, D]`
//!                         Prepends the class token to each batch item's sequence.
//!
//! `dinov2_add_pos_embed`: `[B, N, D]` + pos_embed `[N, D]` → `[B, N, D]`
//!                         Broadcasts positional embedding over batch.
//!
//! `dinov2_remove_cls`:    `[B, N+1, D]` → `[B, N, D]`
//!                         Strips the class token (position 0) from each sequence.
//!
//! These three operations pipeline as:
//!   patches `[B, N, D]`
//!     → cat_cls → `[B, N+1, D]`
//!     → add_pos_embed (N+1 tokens) → `[B, N+1, D]`
//!     → (N ViT blocks)
//!     → remove_cls → `[B, N, D]`

#![allow(non_snake_case)]

use std::sync::Arc;

use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

// ── dinov2_cat_cls ────────────────────────────────────────────────────────────

/// Prepends a shared class token `[D]` to a batch of sequences `[B, N, D]`.
///
/// Output: `out[b, 0, d]  = cls[d]`
///         `out[b, i+1, d] = in[b, i, d]`  for `i ∈ [0, N)`.
///
/// Grid: `[B × (N+1), 1, 1]`; one CTA per `(b, pos)`.
///
/// `BLOCK_D` must be power of 2; `embed_dim` is the actual D (≤ BLOCK_D).
#[kernel]
pub fn dinov2_cat_cls<T: Triton, const BLOCK_D: i32>(
    in_ptr:    T::Pointer<f32>,  // [B, N, D]
    cls_ptr:   T::Pointer<f32>,  // [D]  — class-token parameter
    out_ptr:   T::Pointer<f32>,  // [B, N+1, D]
    n_seq:     i32,              // N
    embed_dim: i32,              // actual D (≤ BLOCK_D)
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid   = T::program_id(Axis::X);  // [0, B * (N+1))
    let d     = T::arange(0, BLOCK_D);
    let mask  = d.lt(embed_dim);
    let n_out = n_seq + 1;

    let b   = pid / n_out;
    let pos = pid - b * n_out;

    let out_base = (b * n_out + pos) * embed_dim;
    let zeros    = T::zeros::<f32>(&[BLOCK_D]);

    if pos == 0 {
        // Copy class token
        let x = T::load(cls_ptr.add_offsets(d), Some(mask), Some(zeros), &[], None, None, None, false);
        T::store(out_ptr.add_offsets(d + out_base), x, Some(mask), &[], None, None);
    } else {
        // Copy input token at seq position pos-1
        let in_base = (b * n_seq + pos - 1) * embed_dim;
        let x = T::load(in_ptr.add_offsets(d + in_base), Some(mask), Some(zeros), &[], None, None, None, false);
        T::store(out_ptr.add_offsets(d + out_base), x, Some(mask), &[], None, None);
    }
}

// ── dinov2_cat_cls_backward ───────────────────────────────────────────────────

/// Backward of `dinov2_cat_cls`.
///
/// Splits the gradient `[B, N+1, D]` back into:
/// - `d_in[b, i, d] = grad[b, i+1, d]`
/// - `d_cls[d]` accumulated via atomic add across all batch items from `grad[b, 0, d]`.
///
/// Grid: `[B × (N+1), 1, 1]`.
#[kernel]
pub fn dinov2_cat_cls_backward<T: Triton, const BLOCK_D: i32>(
    grad_out_ptr: T::Pointer<f32>,  // [B, N+1, D]
    d_in_ptr:     T::Pointer<f32>,  // [B, N, D]
    d_cls_ptr:    T::Pointer<f32>,  // [D]   — atomic add
    n_seq:        i32,
    embed_dim:    i32,              // actual D (≤ BLOCK_D)
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid   = T::program_id(Axis::X);
    let d     = T::arange(0, BLOCK_D);
    let mask  = d.lt(embed_dim);
    let n_out = n_seq + 1;

    let b   = pid / n_out;
    let pos = pid - b * n_out;

    let grad_base = (b * n_out + pos) * embed_dim;
    let zeros     = T::zeros::<f32>(&[BLOCK_D]);
    let gx = T::load(grad_out_ptr.add_offsets(d + grad_base), Some(mask), Some(zeros), &[], None, None, None, false);

    if pos == 0 {
        // Accumulate class-token gradient
        T::atomic_add(d_cls_ptr.add_offsets(d), gx, Some(mask), None, None);
    } else {
        // Write patch-token gradient
        let in_base = (b * n_seq + pos - 1) * embed_dim;
        T::store(d_in_ptr.add_offsets(d + in_base), gx, Some(mask), &[], None, None);
    }
}

// ── dinov2_add_pos_embed ──────────────────────────────────────────────────────

/// Broadcasts positional embedding `[N, D]` onto `[B, N, D]` in-place style.
///
/// `out[b, n, d] = in[b, n, d] + pos[n, d]`
///
/// Grid: `[B × N, 1, 1]`.
#[kernel]
pub fn dinov2_add_pos_embed<T: Triton, const BLOCK_D: i32>(
    in_ptr:    T::Pointer<f32>,  // [B, N, D]
    pos_ptr:   T::Pointer<f32>,  // [N, D]  — positional embedding parameter
    out_ptr:   T::Pointer<f32>,  // [B, N, D]
    n_seq:     i32,
    embed_dim: i32,              // actual D (≤ BLOCK_D)
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid  = T::program_id(Axis::X);  // [0, B * N)
    let d    = T::arange(0, BLOCK_D);
    let mask = d.lt(embed_dim);

    let b = pid / n_seq;
    let n = pid - b * n_seq;

    let tok_base = (b * n_seq + n) * embed_dim;
    let pos_base = n * embed_dim;

    let zeros = T::zeros::<f32>(&[BLOCK_D]);
    let x  = T::load(in_ptr.add_offsets(d + tok_base),  Some(mask), Some(zeros), &[], None, None, None, false);
    let pe = T::load(pos_ptr.add_offsets(d + pos_base), Some(mask), Some(zeros), &[], None, None, None, false);
    T::store(out_ptr.add_offsets(d + tok_base), x + pe, Some(mask), &[], None, None);
}

// ── dinov2_add_pos_embed_backward ─────────────────────────────────────────────

/// Backward of `dinov2_add_pos_embed`.
///
/// `d_in[b, n, d] = grad[b, n, d]`
/// `d_pos[n, d]`  accumulated via atomic add over batch.
///
/// Grid: `[B × N, 1, 1]`.
#[kernel]
pub fn dinov2_add_pos_embed_backward<T: Triton, const BLOCK_D: i32>(
    grad_ptr:  T::Pointer<f32>,  // [B, N, D]
    d_in_ptr:  T::Pointer<f32>,  // [B, N, D]
    d_pos_ptr: T::Pointer<f32>,  // [N, D]   — atomic add
    n_seq:     i32,
    embed_dim: i32,              // actual D (≤ BLOCK_D)
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid  = T::program_id(Axis::X);
    let d    = T::arange(0, BLOCK_D);
    let mask = d.lt(embed_dim);

    let b = pid / n_seq;
    let n = pid - b * n_seq;

    let tok_base = (b * n_seq + n) * embed_dim;
    let pos_base = n * embed_dim;

    let zeros = T::zeros::<f32>(&[BLOCK_D]);
    let g = T::load(grad_ptr.add_offsets(d + tok_base), Some(mask), Some(zeros), &[], None, None, None, false);
    T::store(d_in_ptr.add_offsets(d + tok_base), g, Some(mask), &[], None, None);
    T::atomic_add(d_pos_ptr.add_offsets(d + pos_base), g, Some(mask), None, None);
}

// ── dinov2_remove_cls ─────────────────────────────────────────────────────────

/// Strips the class token (position 0) from `[B, N+1, D]` → `[B, N, D]`.
///
/// `out[b, n, d] = in[b, n+1, d]`
///
/// Grid: `[B × N, 1, 1]`.
#[kernel]
pub fn dinov2_remove_cls<T: Triton, const BLOCK_D: i32>(
    in_ptr:    T::Pointer<f32>,  // [B, N+1, D]
    out_ptr:   T::Pointer<f32>,  // [B, N, D]
    n_seq:     i32,              // N (patch count, NOT N+1)
    embed_dim: i32,              // actual D (≤ BLOCK_D)
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid   = T::program_id(Axis::X);  // [0, B * N)
    let d     = T::arange(0, BLOCK_D);
    let mask  = d.lt(embed_dim);
    let n_out = n_seq + 1;

    let b = pid / n_seq;
    let n = pid - b * n_seq;

    // Source position n+1 (skip class token at 0)
    let in_base  = (b * n_out + n + 1) * embed_dim;
    let out_base = (b * n_seq + n) * embed_dim;

    let zeros = T::zeros::<f32>(&[BLOCK_D]);
    let x = T::load(in_ptr.add_offsets(d + in_base), Some(mask), Some(zeros), &[], None, None, None, false);
    T::store(out_ptr.add_offsets(d + out_base), x, Some(mask), &[], None, None);
}

// ── dinov2_remove_cls_backward ────────────────────────────────────────────────

/// Backward of `dinov2_remove_cls`.
///
/// Gradient passes through patch positions only; class-token gradient is zero.
/// Grid: `[B × N, 1, 1]`.
#[kernel]
pub fn dinov2_remove_cls_backward<T: Triton, const BLOCK_D: i32>(
    grad_ptr:  T::Pointer<f32>,  // [B, N, D]
    out_ptr:   T::Pointer<f32>,  // [B, N+1, D]   (cls slot left as-is / zeroed by runtime)
    n_seq:     i32,
    embed_dim: i32,              // actual D (≤ BLOCK_D)
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid   = T::program_id(Axis::X);
    let d     = T::arange(0, BLOCK_D);
    let mask  = d.lt(embed_dim);
    let n_out = n_seq + 1;

    let b = pid / n_seq;
    let n = pid - b * n_seq;

    let grad_base = (b * n_seq + n) * embed_dim;
    let out_base  = (b * n_out + n + 1) * embed_dim;

    let zeros = T::zeros::<f32>(&[BLOCK_D]);
    let g = T::load(grad_ptr.add_offsets(d + grad_base), Some(mask), Some(zeros), &[], None, None, None, false);
    T::store(out_ptr.add_offsets(d + out_base), g, Some(mask), &[], None, None);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn next_pow2(n: usize) -> usize {
    if n == 0 { return 1; }
    let mut p = 1usize;
    while p < n { p <<= 1; }
    p
}

// ── RuntimeOps ───────────────────────────────────────────────────────────────

pub struct Dinov2CatClsRuntimeOp {
    fwd:       Dinov2CatCls,
    bwd:       Dinov2CatClsBackward,
    embed_dim: usize,
}

impl Dinov2CatClsRuntimeOp {
    pub fn new(embed_dim: i32) -> Self {
        let block_d = next_pow2(embed_dim as usize) as i32;
        Self {
            fwd:       Dinov2CatCls::new(block_d),
            bwd:       Dinov2CatClsBackward::new(block_d),
            embed_dim: embed_dim as usize,
        }
    }

    pub fn kernel_name(&self)     -> &str { self.fwd.name }
    pub fn forward_source(&self)  -> &str { &self.fwd.source }
    pub fn backward_source(&self) -> &str { &self.bwd.source }
}

impl teeny_core::model::RuntimeOp for Dinov2CatClsRuntimeOp {
    fn n_activation_inputs(&self) -> usize { 1 }

    // cls_token parameter shape: [embed_dim]
    fn param_shapes(&self, input_shapes: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        let d = *input_shapes[0].last().unwrap();
        vec![vec![d]]
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // inputs[0].1 = [B, N, D]
        let n_seq = inputs[0].1[1] as i32;

        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(n_seq);
        visitor.visit_i32(self.embed_dim as i32);
    }

    fn block(&self) -> [u32; 3] { [1, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [B, N+1, D]
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
        grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n_seq = inputs[0].1[1] as i32;

        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_ptr(grad_params[0]);
        visitor.visit_i32(n_seq);
        visitor.visit_i32(self.embed_dim as i32);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [1, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [B, N+1, D]; bwd grid = (B*(N+1), 1, 1)
        [(output_shape[0] * output_shape[1]) as u32, 1, 1]
    }
}

pub struct Dinov2AddPosEmbedRuntimeOp {
    fwd:       Dinov2AddPosEmbed,
    bwd:       Dinov2AddPosEmbedBackward,
    embed_dim: usize,
}

impl Dinov2AddPosEmbedRuntimeOp {
    pub fn new(embed_dim: i32) -> Self {
        let block_d = next_pow2(embed_dim as usize) as i32;
        Self {
            fwd:       Dinov2AddPosEmbed::new(block_d),
            bwd:       Dinov2AddPosEmbedBackward::new(block_d),
            embed_dim: embed_dim as usize,
        }
    }

    pub fn kernel_name(&self)     -> &str { self.fwd.name }
    pub fn forward_source(&self)  -> &str { &self.fwd.source }
    pub fn backward_source(&self) -> &str { &self.bwd.source }
}

impl teeny_core::model::RuntimeOp for Dinov2AddPosEmbedRuntimeOp {
    fn n_activation_inputs(&self) -> usize { 1 }

    // pos_embed parameter: [N, D]
    fn param_shapes(&self, input_shapes: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![input_shapes[0][1], input_shapes[0][2]]]
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n_seq = inputs[0].1[1] as i32;

        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(n_seq);
        visitor.visit_i32(self.embed_dim as i32);
    }

    fn block(&self) -> [u32; 3] { [1, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
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
        grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n_seq = inputs[0].1[1] as i32;

        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_ptr(grad_params[0]);
        visitor.visit_i32(n_seq);
        visitor.visit_i32(self.embed_dim as i32);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [1, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        [(output_shape[0] * output_shape[1]) as u32, 1, 1]
    }
}

pub struct Dinov2RemoveClsRuntimeOp {
    fwd:       Dinov2RemoveCls,
    bwd:       Dinov2RemoveClsBackward,
    embed_dim: usize,
}

impl Dinov2RemoveClsRuntimeOp {
    pub fn new(embed_dim: i32) -> Self {
        let block_d = next_pow2(embed_dim as usize) as i32;
        Self {
            fwd:       Dinov2RemoveCls::new(block_d),
            bwd:       Dinov2RemoveClsBackward::new(block_d),
            embed_dim: embed_dim as usize,
        }
    }

    pub fn kernel_name(&self)     -> &str { self.fwd.name }
    pub fn forward_source(&self)  -> &str { &self.fwd.source }
    pub fn backward_source(&self) -> &str { &self.bwd.source }
}

impl teeny_core::model::RuntimeOp for Dinov2RemoveClsRuntimeOp {
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
        // inputs[0].1 = [B, N+1, D]; output.1 = [B, N, D]
        let n_seq = (inputs[0].1[1] - 1) as i32;

        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n_seq);
        visitor.visit_i32(self.embed_dim as i32);
    }

    fn block(&self) -> [u32; 3] { [1, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
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
        let n_seq = (inputs[0].1[1] - 1) as i32;

        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_i32(n_seq);
        visitor.visit_i32(self.embed_dim as i32);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [1, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, input_shapes: &[&[usize]], _: &[usize]) -> [u32; 3] {
        // grad goes back to [B, N+1, D]
        [(input_shapes[0][0] * input_shapes[0][1]) as u32, 1, 1]
    }
}

// ── CustomOps ─────────────────────────────────────────────────────────────────

use std::any::Any;
use teeny_core::{
    graph::{CustomOp, Shape},
    model::RuntimeOp,
};

/// Graph node: `[B, N, D]` + cls_token param `[D]` → `[B, N+1, D]`.
pub struct Dinov2CatClsOp {
    inner:     Arc<Dinov2CatClsRuntimeOp>,
    embed_dim: usize,
}

impl Dinov2CatClsOp {
    pub fn new(embed_dim: i32) -> Self {
        Self {
            inner:     Arc::new(Dinov2CatClsRuntimeOp::new(embed_dim)),
            embed_dim: embed_dim as usize,
        }
    }
}

impl CustomOp for Dinov2CatClsOp {
    fn name(&self) -> &str { "dinov2_cat_cls" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // input: [B, N, D] → output: [B, N+1, D]
        let b = input_shapes[0][0];
        let n = input_shapes[0][1].map(|nv| nv + 1);
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

/// Graph node: `[B, N, D]` + pos_embed param `[N, D]` → `[B, N, D]`.
pub struct Dinov2AddPosEmbedOp {
    inner: Arc<Dinov2AddPosEmbedRuntimeOp>,
}

impl Dinov2AddPosEmbedOp {
    pub fn new(embed_dim: i32) -> Self {
        Self { inner: Arc::new(Dinov2AddPosEmbedRuntimeOp::new(embed_dim)) }
    }
}

impl CustomOp for Dinov2AddPosEmbedOp {
    fn name(&self) -> &str { "dinov2_add_pos_embed" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].to_vec()
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

/// Graph node: `[B, N+1, D]` → `[B, N, D]` (remove class token).
pub struct Dinov2RemoveClsOp {
    inner:     Arc<Dinov2RemoveClsRuntimeOp>,
    embed_dim: usize,
}

impl Dinov2RemoveClsOp {
    pub fn new(embed_dim: i32) -> Self {
        Self {
            inner:     Arc::new(Dinov2RemoveClsRuntimeOp::new(embed_dim)),
            embed_dim: embed_dim as usize,
        }
    }
}

impl CustomOp for Dinov2RemoveClsOp {
    fn name(&self) -> &str { "dinov2_remove_cls" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // input: [B, N+1, D] → output: [B, N, D]
        let b = input_shapes[0][0];
        let n = input_shapes[0][1].map(|nv| nv - 1);
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
