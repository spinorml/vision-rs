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


//! Sequence-layout helper Triton kernels for the DETR decoder.
//!
//! `pack_heads`:  `[B, S, n_heads * HD] → [BH, S, HD]` — scatter heads to batch dim.
//! `seq_cat2`:    `[B, Sa, D] + [B, Sb, D] → [B, Sa+Sb, D]` — concat along seq dim.
//!
//! Both have corresponding backward kernels.

#![allow(non_snake_case)]

use std::sync::Arc;
use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};
use teeny_core::graph::Shape;

// ── pack_heads forward ────────────────────────────────────────────────────────
//
// Input:  [B, S, n_heads * HD]
// Output: [B * n_heads, S, HD]  (= [BH, S, HD])
// Grid:   (BH * S, 1, 1)
//
// BLOCK_HD must be a power of 2 (Triton constraint); the actual head_dim may be
// smaller.  Elements past head_dim are masked out to avoid out-of-bounds access.

#[kernel]
pub fn pack_heads<T: Triton, const BLOCK_HD: i32>(
    inp_ptr:  T::Pointer<f32>,  // [B, S, n_heads * head_dim]
    out_ptr:  T::Pointer<f32>,  // [BH, S, head_dim]
    n_heads:  i32,
    s:        i32,
    head_dim: i32,              // actual HD (≤ BLOCK_HD)
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid  = T::program_id(Axis::X); // [0, BH*S)
    let hd   = T::arange(0, BLOCK_HD);
    let mask = hd.lt(head_dim);
    let d    = n_heads * head_dim;

    let bh = pid / s;
    let si = pid - bh * s;
    let b  = bh / n_heads;
    let h  = bh - b * n_heads;

    let in_base  = (b * s + si) * d + h * head_dim;
    let out_base = (bh * s + si) * head_dim;

    let zeros = T::zeros::<f32>(&[BLOCK_HD]);
    let x = T::load(inp_ptr.add_offsets(hd + in_base), Some(mask), Some(zeros), &[], None, None, None, false);
    T::store(out_ptr.add_offsets(hd + out_base), x, Some(mask), &[], None, None);
}

// ── pack_heads backward ───────────────────────────────────────────────────────
//
// Gradient of pack_heads is an unpack (same element-wise mapping, transposed).
// grad_out: [BH, S, HD] → grad_inp: [B, S, n_heads * HD]
// Grid: same (BH * S, 1, 1).

#[kernel]
pub fn unpack_heads<T: Triton, const BLOCK_HD: i32>(
    grad_out_ptr: T::Pointer<f32>, // [BH, S, head_dim]
    grad_inp_ptr: T::Pointer<f32>, // [B, S, n_heads * head_dim]
    n_heads:      i32,
    s:            i32,
    head_dim:     i32,             // actual HD (≤ BLOCK_HD)
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid  = T::program_id(Axis::X);
    let hd   = T::arange(0, BLOCK_HD);
    let mask = hd.lt(head_dim);
    let d    = n_heads * head_dim;

    let bh = pid / s;
    let si = pid - bh * s;
    let b  = bh / n_heads;
    let h  = bh - b * n_heads;

    let gout_base = (bh * s + si) * head_dim;
    let ginp_base = (b * s + si) * d + h * head_dim;

    let zeros = T::zeros::<f32>(&[BLOCK_HD]);
    let dx = T::load(grad_out_ptr.add_offsets(hd + gout_base), Some(mask), Some(zeros), &[], None, None, None, false);
    T::store(grad_inp_ptr.add_offsets(hd + ginp_base), dx, Some(mask), &[], None, None);
}

// ── seq_cat2 forward ──────────────────────────────────────────────────────────
//
// Concatenate [B, Sa, D] and [B, Sb, D] along dim 1.
// Output: [B, Sa + Sb, D].
// Grid: (B * (Sa + Sb), 1, 1)

#[kernel]
pub fn seq_cat2<T: Triton, const D: i32>(
    a_ptr:   T::Pointer<f32>,  // [B, Sa, D]
    b_ptr:   T::Pointer<f32>,  // [B, Sb, D]
    out_ptr: T::Pointer<f32>,  // [B, Sa+Sb, D]
    sa: i32,
    sb: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let d   = T::arange(0, D);
    let s_total = sa + sb;

    let b = pid / s_total;
    let s = pid - b * s_total;

    let out_base = (b * s_total + s) * D;

    if s < sa {
        let x = T::load(a_ptr.add_offsets(d + (b * sa + s) * D), None, None, &[], None, None, None, false);
        T::store(out_ptr.add_offsets(d + out_base), x, None, &[], None, None);
    } else {
        let sb_idx = s - sa;
        let x = T::load(b_ptr.add_offsets(d + (b * sb + sb_idx) * D), None, None, &[], None, None, None, false);
        T::store(out_ptr.add_offsets(d + out_base), x, None, &[], None, None);
    }
}

// ── seq_cat2 backward ─────────────────────────────────────────────────────────
//
// Split grad_out back into the two source tensors.
// Grid: same as forward (B * (Sa + Sb), 1, 1).

#[kernel]
pub fn seq_split2<T: Triton, const D: i32>(
    grad_out_ptr: T::Pointer<f32>, // [B, Sa+Sb, D]
    grad_a_ptr:   T::Pointer<f32>, // [B, Sa, D]
    grad_b_ptr:   T::Pointer<f32>, // [B, Sb, D]
    sa: i32,
    sb: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let d   = T::arange(0, D);
    let s_total = sa + sb;

    let b = pid / s_total;
    let s = pid - b * s_total;

    let gout_base = (b * s_total + s) * D;
    let gx = T::load(grad_out_ptr.add_offsets(d + gout_base), None, None, &[], None, None, None, false);

    if s < sa {
        T::store(grad_a_ptr.add_offsets(d + (b * sa + s) * D), gx, None, &[], None, None);
    } else {
        let sb_idx = s - sa;
        T::store(grad_b_ptr.add_offsets(d + (b * sb + sb_idx) * D), gx, None, &[], None, None);
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn next_pow2(n: usize) -> usize {
    if n == 0 { return 1; }
    let mut p = 1usize;
    while p < n { p <<= 1; }
    p
}

// ── RuntimeOp: PackHeadsRuntimeOp ─────────────────────────────────────────────

pub struct PackHeadsRuntimeOp {
    fwd:      PackHeads,
    bwd:      UnpackHeads,
    head_dim: usize,
    n_heads:  usize,
}

impl PackHeadsRuntimeOp {
    pub fn new(head_dim: i32, n_heads: usize) -> Self {
        let block_hd = next_pow2(head_dim as usize) as i32;
        Self {
            fwd:      PackHeads::new(block_hd),
            bwd:      UnpackHeads::new(block_hd),
            head_dim: head_dim as usize,
            n_heads,
        }
    }
    pub fn kernel_name(&self)     -> &str { self.fwd.name }
    pub fn forward_source(&self)  -> &str { &self.fwd.source }
    pub fn backward_source(&self) -> &str { &self.bwd.source }
}

impl teeny_core::model::RuntimeOp for PackHeadsRuntimeOp {
    fn n_activation_inputs(&self) -> usize { 1 }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { Vec::new() }

    fn compute_concrete_output_shape(&self, input_shapes: &[&[usize]], _resolved: &[usize]) -> Vec<usize> {
        // input: [B, S, n_heads*HD] → output: [BH, S, HD]
        let b  = input_shapes[0][0];
        let s  = input_shapes[0][1];
        vec![b * self.n_heads, s, self.head_dim]
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
        // inputs[0]: [B, S, n_heads * HD]
        let s = inputs[0].1[1] as i32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.n_heads as i32);
        visitor.visit_i32(s);
        visitor.visit_i32(self.head_dim as i32);
    }

    fn block(&self) -> [u32; 3] { [1, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [BH, S, HD]; grid = (BH * S, 1, 1)
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
        let bh = inputs[0].1[0] * self.n_heads;  // note: inputs[0] is original input
        let s  = inputs[0].1[1] as i32;
        // bwd = unpack_heads: grad_out[BH,S,HD] → grad_inp[B,S,D]
        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_i32(self.n_heads as i32);
        visitor.visit_i32(s);
        visitor.visit_i32(self.head_dim as i32);
        let _ = bh;
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [1, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, input_shapes: &[&[usize]], _: &[usize]) -> [u32; 3] {
        // grad_out is [BH, S, HD]; grid = (BH * S, 1, 1)
        let b  = input_shapes[0][0] * self.n_heads;
        let s  = input_shapes[0][1];
        [(b * s) as u32, 1, 1]
    }
}

// ── RuntimeOp: SeqCat2RuntimeOp ───────────────────────────────────────────────

pub struct SeqCat2RuntimeOp {
    fwd: SeqCat2,
    bwd: SeqSplit2,
    #[allow(dead_code)]
    d:   usize,
}

impl SeqCat2RuntimeOp {
    pub fn new(d: i32) -> Self {
        Self {
            fwd: SeqCat2::new(d),
            bwd: SeqSplit2::new(d),
            d:   d as usize,
        }
    }
    pub fn kernel_name(&self)     -> &str { self.fwd.name }
    pub fn forward_source(&self)  -> &str { &self.fwd.source }
    pub fn backward_source(&self) -> &str { &self.bwd.source }
}

impl teeny_core::model::RuntimeOp for SeqCat2RuntimeOp {
    fn n_activation_inputs(&self) -> usize { 2 }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { Vec::new() }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // inputs[0]: [B, Sa, D]; inputs[1]: [B, Sb, D]
        let sa = inputs[0].1[1] as i32;
        let sb = inputs[1].1[1] as i32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(sa);
        visitor.visit_i32(sb);
    }

    fn block(&self) -> [u32; 3] { [1, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [B, Sa+Sb, D]; grid = (B * (Sa+Sb), 1, 1)
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
        let sa = inputs[0].1[1] as i32;
        let sb = inputs[1].1[1] as i32;
        // bwd = seq_split2
        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_ptr(grad_inputs[1]);
        visitor.visit_i32(sa);
        visitor.visit_i32(sb);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [1, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, input_shapes: &[&[usize]], _: &[usize]) -> [u32; 3] {
        // grad_out is [B, Sa+Sb, D]
        let b  = input_shapes[0][0];
        let sa = input_shapes[0][1];
        let sb = input_shapes[1][1];
        [(b * (sa + sb)) as u32, 1, 1]
    }
}

// ── CustomOp: PackHeadsOp ─────────────────────────────────────────────────────

use std::any::Any;
use teeny_core::graph::CustomOp;

/// Graph node: `[B, S, n_heads * HD]` → `[BH, S, HD]`.
pub struct PackHeadsOp {
    inner:    Arc<PackHeadsRuntimeOp>,
    head_dim: usize,
    n_heads:  usize,
}

impl PackHeadsOp {
    pub fn new(head_dim: i32, n_heads: usize) -> Self {
        Self {
            inner:    Arc::new(PackHeadsRuntimeOp::new(head_dim, n_heads)),
            head_dim: head_dim as usize,
            n_heads,
        }
    }
}

impl CustomOp for PackHeadsOp {
    fn name(&self) -> &str { "pack_heads" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let b  = input_shapes[0][0];
        let s  = input_shapes[0][1];
        let bh = b.map(|bv| bv * self.n_heads);
        vec![bh, s, Some(self.head_dim)]
    }

    fn as_any(&self) -> &dyn Any { self }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn teeny_core::model::RuntimeOp>)> {
        Some((
            self.inner.kernel_name().to_string(),
            self.inner.forward_source().to_string(),
            "entry_point".to_string(),
            Arc::clone(&self.inner) as Arc<dyn teeny_core::model::RuntimeOp>,
        ))
    }

    fn lower_backward_source(&self) -> String {
        self.inner.backward_source().to_string()
    }
}

// ── CustomOp: SeqCat2Op ───────────────────────────────────────────────────────

/// Graph node: `[B, Sa, D]` + `[B, Sb, D]` → `[B, Sa+Sb, D]`.
pub struct SeqCat2Op {
    inner: Arc<SeqCat2RuntimeOp>,
    d:     usize,
}

impl SeqCat2Op {
    pub fn new(d: i32) -> Self {
        Self { inner: Arc::new(SeqCat2RuntimeOp::new(d)), d: d as usize }
    }
}

impl CustomOp for SeqCat2Op {
    fn name(&self) -> &str { "seq_cat2" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let b       = input_shapes[0][0];
        let sa      = input_shapes[0][1];
        let sb      = input_shapes[1][1];
        let s_total = sa.zip(sb).map(|(a, b)| a + b);
        vec![b, s_total, Some(self.d)]
    }

    fn as_any(&self) -> &dyn Any { self }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn teeny_core::model::RuntimeOp>)> {
        Some((
            self.inner.kernel_name().to_string(),
            self.inner.forward_source().to_string(),
            "entry_point".to_string(),
            Arc::clone(&self.inner) as Arc<dyn teeny_core::model::RuntimeOp>,
        ))
    }

    fn lower_backward_source(&self) -> String {
        self.inner.backward_source().to_string()
    }
}
