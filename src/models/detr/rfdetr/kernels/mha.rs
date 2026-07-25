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


//! Multi-Head Attention (no RoPE) kernels for DINOv2 ViT blocks.
//!
//! Wraps Flash Attention 2 with DINOv2-specific QKV packing/unpacking:
//!
//! Forward:
//!   1. `dinov2_pack_qkv`:   `[B, N, 3*H*HD]` → packed `[3*BH, N, HD]`
//!   2. Flash Attention 2:    Q, K, V from packed buffer → O `[BH, N, HD]`
//!   3. `dinov2_unpack_attn`: `[BH, N, HD]` → `[B, N, H*HD]`
//!
//! Backward is the transpose of each step.
//!
//! Layout notes:
//!   - QKV linear output: `qkv[b, n, s*H*HD + h*HD + d]` (s=0:Q, 1:K, 2:V)
//!   - Packed buffer:      `packed[s, bh, n, d]` (bh = b*H + h)
//!   - Attention output:   `attn[bh, n, d]`
//!   - Unpacked output:    `out[b, n, h*HD + d]`

#![allow(non_snake_case)]

use core::ffi::c_void;
use std::sync::Arc;

use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison, Tensor},
    *,
};

use super::super::super::super::yolo::kernels::attention::flash_attn2::{
    FlashAttention2BackwardDq, FlashAttention2Forward,
};

// ── dinov2_pack_qkv ───────────────────────────────────────────────────────────

/// Packs the combined QKV tensor from a linear projection into Flash Attention 2
/// input format.
///
/// Input:  `qkv_ptr`    — `[B, N, 3*num_heads*HEAD_DIM]` row-major.
///           Channel layout: `qkv[b, n, s*num_heads*HD + h*HD + d]`, s=0:Q, 1:K, 2:V.
/// Output: `packed_ptr` — `[3*BH, N, HEAD_DIM]` where `BH = B * num_heads`.
///           Layout: `packed[s*BH + bh, n, d]`.
///
/// Grid:  `[3 * BH * N, 1, 1]` — one CTA per (section, batch-head, token).
/// Block: `[HEAD_DIM, 1, 1]` → Triton emits 128 threads (4 warps).
#[kernel]
pub fn dinov2_pack_qkv<T: Triton, const HEAD_DIM: i32>(
    qkv_ptr:    T::Pointer<f32>, // [B, N, 3*num_heads*HD]
    packed_ptr: T::Pointer<f32>, // [3*BH, N, HD]
    n_ctx:      i32,             // N: sequence length
    num_heads:  i32,             // H: number of attention heads
    bh_total:   i32,             // BH = B * num_heads
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X); // [0, 3 * BH * N)
    let d   = T::arange(0, HEAD_DIM);

    // Decompose pid into (s, bh, n)
    let s  = pid / (bh_total * n_ctx);               // section: 0=Q, 1=K, 2=V
    let bh = (pid / n_ctx) - s * bh_total;           // batch-head index [0, BH)
    let n  = pid - (s * bh_total + bh) * n_ctx;      // token index [0, N)
    let b  = bh / num_heads;                          // batch index
    let h  = bh - b * num_heads;                      // head index

    // Source: qkv[b, n, s*num_heads*HD + h*HD + d]
    let src_base = b * n_ctx * 3 * num_heads * HEAD_DIM
                 + n * 3 * num_heads * HEAD_DIM
                 + s * num_heads * HEAD_DIM
                 + h * HEAD_DIM;
    let x = T::load(qkv_ptr.add_offsets(d + src_base), None, None, &[], None, None, None, false);

    // Destination: packed[(s*BH + bh), n, d]
    let dst_base = (s * bh_total + bh) * n_ctx * HEAD_DIM + n * HEAD_DIM;
    T::store(packed_ptr.add_offsets(d + dst_base), x, None, &[], None, None);
}

// ── dinov2_pack_qkv_backward ──────────────────────────────────────────────────

/// Backward of `dinov2_pack_qkv`: scatters gradient from packed layout back
/// to the QKV linear format.
///
/// Each (s, bh, n, d) in the packed buffer maps to a unique (b, n, s, h, d)
/// in the QKV buffer, so regular stores (no atomics) are safe.
///
/// Grid / Block match `dinov2_pack_qkv`.
#[kernel]
pub fn dinov2_pack_qkv_backward<T: Triton, const HEAD_DIM: i32>(
    d_packed_ptr: T::Pointer<f32>, // [3*BH, N, HD]  — gradient input
    d_qkv_ptr:    T::Pointer<f32>, // [B, N, 3*H*HD] — gradient output
    n_ctx:        i32,
    num_heads:    i32,
    bh_total:     i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let d   = T::arange(0, HEAD_DIM);

    let s  = pid / (bh_total * n_ctx);
    let bh = (pid / n_ctx) - s * bh_total;
    let n  = pid - (s * bh_total + bh) * n_ctx;
    let b  = bh / num_heads;
    let h  = bh - b * num_heads;

    // Read gradient from packed buffer
    let src_base = (s * bh_total + bh) * n_ctx * HEAD_DIM + n * HEAD_DIM;
    let dx = T::load(d_packed_ptr.add_offsets(d + src_base), None, None, &[], None, None, None, false);

    // Write back to QKV gradient
    let dst_base = b * n_ctx * 3 * num_heads * HEAD_DIM
                 + n * 3 * num_heads * HEAD_DIM
                 + s * num_heads * HEAD_DIM
                 + h * HEAD_DIM;
    T::store(d_qkv_ptr.add_offsets(d + dst_base), dx, None, &[], None, None);
}

// ── dinov2_unpack_attn ────────────────────────────────────────────────────────

/// Unpacks Flash Attention 2 output `[BH, N, HD]` to `[B, N, H*HD]`.
///
/// Inverse of the Q/K/V permutation applied to the output O.
///
/// Grid:  `[BH * N, 1, 1]`.
/// Block: `[HEAD_DIM, 1, 1]`.
#[kernel]
pub fn dinov2_unpack_attn<T: Triton, const HEAD_DIM: i32>(
    attn_ptr: T::Pointer<f32>,   // [BH, N, HD]
    out_ptr:  T::Pointer<f32>,   // [B, N, H*HD]
    n_ctx:     i32,
    num_heads: i32,
    _bh_total: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X); // [0, BH * N)
    let d   = T::arange(0, HEAD_DIM);

    let bh = pid / n_ctx;
    let n  = pid - bh * n_ctx;
    let b  = bh / num_heads;
    let h  = bh - b * num_heads;

    // Source: attn[bh, n, d]
    let src_base = bh * n_ctx * HEAD_DIM + n * HEAD_DIM;
    let x = T::load(attn_ptr.add_offsets(d + src_base), None, None, &[], None, None, None, false);

    // Destination: out[b, n, h*HD + d]
    let dst_base = b * n_ctx * num_heads * HEAD_DIM
                 + n * num_heads * HEAD_DIM
                 + h * HEAD_DIM;
    T::store(out_ptr.add_offsets(d + dst_base), x, None, &[], None, None);
}

// ── dinov2_unpack_attn_backward ───────────────────────────────────────────────

/// Backward of `dinov2_unpack_attn`: maps gradient from `[B, N, H*HD]` back
/// to `[BH, N, HD]`.  Regular stores (unique mapping).
#[kernel]
pub fn dinov2_unpack_attn_backward<T: Triton, const HEAD_DIM: i32>(
    d_out_ptr:  T::Pointer<f32>, // [B, N, H*HD] — gradient input
    d_attn_ptr: T::Pointer<f32>, // [BH, N, HD]  — gradient output
    n_ctx:      i32,
    num_heads:  i32,
    _bh_total:  i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let d   = T::arange(0, HEAD_DIM);

    let bh = pid / n_ctx;
    let n  = pid - bh * n_ctx;
    let b  = bh / num_heads;
    let h  = bh - b * num_heads;

    // Read gradient from unpacked layout
    let src_base = b * n_ctx * num_heads * HEAD_DIM
                 + n * num_heads * HEAD_DIM
                 + h * HEAD_DIM;
    let dx = T::load(d_out_ptr.add_offsets(d + src_base), None, None, &[], None, None, None, false);

    // Write to attn gradient
    let dst_base = bh * n_ctx * HEAD_DIM + n * HEAD_DIM;
    T::store(d_attn_ptr.add_offsets(d + dst_base), dx, None, &[], None, None);
}

// ── RuntimeOp: Dinov2PackQkvRuntimeOp ────────────────────────────────────────

pub struct Dinov2PackQkvRuntimeOp {
    fwd:       Dinov2PackQkv,
    bwd:       Dinov2PackQkvBackward,
    num_heads: usize,
}

impl Dinov2PackQkvRuntimeOp {
    pub fn new(head_dim: i32, num_heads: usize) -> Self {
        Self {
            fwd: Dinov2PackQkv::new(head_dim),
            bwd: Dinov2PackQkvBackward::new(head_dim),
            num_heads,
        }
    }

    pub fn kernel_name(&self)    -> &str { self.fwd.name }
    pub fn forward_source(&self) -> &str { &self.fwd.source }
    pub fn backward_source(&self)-> &str { &self.bwd.source }
}

impl teeny_core::model::RuntimeOp for Dinov2PackQkvRuntimeOp {
    fn n_activation_inputs(&self) -> usize { 1 }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { vec![] }

    fn compute_concrete_output_shape(&self, input_shapes: &[&[usize]], _resolved: &[usize]) -> Vec<usize> {
        // input: [B, N, 3*H*HD] → output: [3*BH, N, HD]
        let b  = input_shapes[0][0];
        let n  = input_shapes[0][1];
        let bh = b * self.num_heads;
        let hd = input_shapes[0][2] / (3 * self.num_heads);
        vec![3 * bh, n, hd]
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
        // inputs[0].1 = [B, N, 3*H*HD]
        let b  = inputs[0].1[0];
        let n  = inputs[0].1[1];
        let bh = b * self.num_heads;

        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n as i32);
        visitor.visit_i32(self.num_heads as i32);
        visitor.visit_i32(bh as i32);
    }

    fn block(&self) -> [u32; 3] { [1, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [3*BH, N, HD]; grid = (3*BH*N, 1, 1)
        [( output_shape[0] * output_shape[1]) as u32, 1, 1]
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
        // inputs[0].1 = [B, N, 3*H*HD]
        let b  = inputs[0].1[0];
        let n  = inputs[0].1[1];
        let bh = b * self.num_heads;

        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_i32(n as i32);
        visitor.visit_i32(self.num_heads as i32);
        visitor.visit_i32(bh as i32);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [1, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [3*BH, N, HD]
        [(output_shape[0] * output_shape[1]) as u32, 1, 1]
    }
}

// ── RuntimeOp: FlashAttn2Dinov2RuntimeOp ─────────────────────────────────────

/// RuntimeOp for Flash Attention 2 in DINOv2 layout.
///
/// Input:  packed QKV buffer `[3*BH, N, HD]`
/// Output: attention output  `[BH, N, HD]`
/// Params: logsumexp scratch  `[BH * N]`
pub struct FlashAttn2Dinov2RuntimeOp {
    fwd:    FlashAttention2Forward,
    bwd_dq: FlashAttention2BackwardDq,
}

impl FlashAttn2Dinov2RuntimeOp {
    pub fn new(head_dim: i32) -> Self {
        Self {
            fwd:    FlashAttention2Forward::new(head_dim),
            bwd_dq: FlashAttention2BackwardDq::new(head_dim),
        }
    }

    pub fn kernel_name(&self)    -> &str { self.fwd.name }
    pub fn forward_source(&self) -> &str { &self.fwd.source }
}

impl teeny_core::model::RuntimeOp for FlashAttn2Dinov2RuntimeOp {
    fn n_activation_inputs(&self) -> usize { 1 }

    fn param_shapes(&self, input_shapes: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        // input_shapes[0] = [3*BH, N, HD]; l_ptr scratch = [BH * N]
        let bh_x3 = input_shapes[0][0];
        let n     = input_shapes[0][1];
        let bh    = bh_x3 / 3;
        vec![vec![bh * n]]
    }

    fn compute_concrete_output_shape(&self, input_shapes: &[&[usize]], _resolved: &[usize]) -> Vec<usize> {
        // input: [3*BH, N, HD] → output: [BH, N, HD]
        let bh = input_shapes[0][0] / 3;
        let n  = input_shapes[0][1];
        let hd = input_shapes[0][2];
        vec![bh, n, hd]
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
        // inputs[0].1 = [3*BH, N, HD]
        let bh  = inputs[0].1[0] / 3;
        let n   = inputs[0].1[1];
        let hd  = inputs[0].1[2];
        let section_elems = bh * n * hd;
        let softmax_scale = 1.0_f32 / (hd as f32).sqrt();

        let base  = inputs[0].0 as *mut f32;
        let q_ptr = base as *mut c_void;
        let k_ptr = unsafe { base.add(section_elems) }        as *mut c_void;
        let v_ptr = unsafe { base.add(2 * section_elems) }    as *mut c_void;

        visitor.visit_ptr(q_ptr);
        visitor.visit_ptr(k_ptr);
        visitor.visit_ptr(v_ptr);
        visitor.visit_ptr(output);
        visitor.visit_ptr(params[0]);
        visitor.visit_i32(n as i32);
        visitor.visit_i32(n as i32);
        visitor.visit_f32(softmax_scale);
        visitor.visit_f32(f32::NEG_INFINITY);
    }

    fn block(&self) -> [u32; 3] { [1, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [BH, N, HD]; FA2 grid = (N, BH, 1)
        [output_shape[1] as u32, output_shape[0] as u32, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        _output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let bh  = inputs[0].1[0] / 3;
        let n   = inputs[0].1[1];
        let hd  = inputs[0].1[2];
        let section_elems = bh * n * hd;
        let softmax_scale = 1.0_f32 / (hd as f32).sqrt();

        let fwd_base = inputs[0].0 as *mut f32;
        let q_ptr = fwd_base as *mut c_void;
        let k_ptr = unsafe { fwd_base.add(section_elems) }     as *mut c_void;
        let v_ptr = unsafe { fwd_base.add(2 * section_elems) } as *mut c_void;

        let dq_ptr = grad_inputs[0];

        // dq backward: (q, k, v, o, do, l, dq, N, scale)
        visitor.visit_ptr(q_ptr);
        visitor.visit_ptr(k_ptr);
        visitor.visit_ptr(v_ptr);
        visitor.visit_ptr(output);
        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(dq_ptr);
        visitor.visit_i32(n as i32);
        visitor.visit_f32(softmax_scale);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [1, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [BH, N, HD]; dQ grid = (N, BH, 1)
        [output_shape[1] as u32, output_shape[0] as u32, 1]
    }
}

// ── RuntimeOp: Dinov2UnpackAttnRuntimeOp ─────────────────────────────────────

pub struct Dinov2UnpackAttnRuntimeOp {
    fwd:       Dinov2UnpackAttn,
    bwd:       Dinov2UnpackAttnBackward,
    num_heads: usize,
}

impl Dinov2UnpackAttnRuntimeOp {
    pub fn new(head_dim: i32, num_heads: usize) -> Self {
        Self {
            fwd: Dinov2UnpackAttn::new(head_dim),
            bwd: Dinov2UnpackAttnBackward::new(head_dim),
            num_heads,
        }
    }

    pub fn kernel_name(&self)    -> &str { self.fwd.name }
    pub fn forward_source(&self) -> &str { &self.fwd.source }
    pub fn backward_source(&self)-> &str { &self.bwd.source }
}

impl teeny_core::model::RuntimeOp for Dinov2UnpackAttnRuntimeOp {
    fn n_activation_inputs(&self) -> usize { 1 }
    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { vec![] }

    fn compute_concrete_output_shape(&self, input_shapes: &[&[usize]], _resolved: &[usize]) -> Vec<usize> {
        // input: [BH, N, HD] → output: [B, N, H*HD]
        let bh = input_shapes[0][0];
        let n  = input_shapes[0][1];
        let hd = input_shapes[0][2];
        let b  = bh / self.num_heads;
        vec![b, n, self.num_heads * hd]
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
        // inputs[0].1 = [BH, N, HD]
        let bh = inputs[0].1[0];
        let n  = inputs[0].1[1];

        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n as i32);
        visitor.visit_i32(self.num_heads as i32);
        visitor.visit_i32(bh as i32);
    }

    fn block(&self) -> [u32; 3] { [1, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [B, N, H*HD]; grid = (BH*N, 1, 1)
        let b = output_shape[0];
        let n = output_shape[1];
        [(b * self.num_heads * n) as u32, 1, 1]
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
        let bh = inputs[0].1[0];
        let n  = inputs[0].1[1];

        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_i32(n as i32);
        visitor.visit_i32(self.num_heads as i32);
        visitor.visit_i32(bh as i32);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [1, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        // grad goes back to [BH, N, HD]; grid = (BH*N, 1, 1)
        let b = output_shape[0];
        let n = output_shape[1];
        [(b * self.num_heads * n) as u32, 1, 1]
    }
}

// ── CustomOp: Dinov2PackQkvOp ─────────────────────────────────────────────────

use std::any::Any;
use teeny_core::{
    graph::{CustomOp, Shape},
    model::RuntimeOp,
};

/// Graph node: packs `[B, N, 3*H*HD]` → `[3*BH, N, HD]`.
pub struct Dinov2PackQkvOp {
    inner:     Arc<Dinov2PackQkvRuntimeOp>,
    num_heads: usize,
    head_dim:  usize,
}

impl Dinov2PackQkvOp {
    pub fn new(head_dim: i32, num_heads: usize) -> Self {
        Self {
            inner:     Arc::new(Dinov2PackQkvRuntimeOp::new(head_dim, num_heads)),
            num_heads,
            head_dim: head_dim as usize,
        }
    }
}

impl CustomOp for Dinov2PackQkvOp {
    fn name(&self) -> &str { "dinov2_pack_qkv" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // input: [B, N, 3*H*HD]
        let b  = input_shapes[0][0];
        let n  = input_shapes[0][1];
        let bh = b.map(|bv| bv * self.num_heads);
        vec![bh.map(|bh| 3 * bh), n, Some(self.head_dim)]
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

// ── CustomOp: FlashAttn2Dinov2Op ─────────────────────────────────────────────

/// Graph node: Flash Attention 2 in DINOv2 layout.
/// Input: `[3*BH, N, HD]`. Output: `[BH, N, HD]`.
pub struct FlashAttn2Dinov2Op {
    inner: Arc<FlashAttn2Dinov2RuntimeOp>,
}

impl FlashAttn2Dinov2Op {
    pub fn new(head_dim: i32) -> Self {
        Self { inner: Arc::new(FlashAttn2Dinov2RuntimeOp::new(head_dim)) }
    }
}

impl CustomOp for FlashAttn2Dinov2Op {
    fn name(&self) -> &str { "flash_attn2_dinov2" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // input: [3*BH, N, HD]
        let bh = input_shapes[0][0].map(|v| v / 3);
        let n  = input_shapes[0][1];
        let hd = input_shapes[0][2];
        vec![bh, n, hd]
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
        self.inner.bwd_dq.source.clone()
    }
}

// ── CustomOp: Dinov2UnpackAttnOp ─────────────────────────────────────────────

/// Graph node: unpacks `[BH, N, HD]` → `[B, N, H*HD]`.
pub struct Dinov2UnpackAttnOp {
    inner:     Arc<Dinov2UnpackAttnRuntimeOp>,
    num_heads: usize,
    head_dim:  usize,
}

impl Dinov2UnpackAttnOp {
    pub fn new(head_dim: i32, num_heads: usize) -> Self {
        Self {
            inner:     Arc::new(Dinov2UnpackAttnRuntimeOp::new(head_dim, num_heads)),
            num_heads,
            head_dim: head_dim as usize,
        }
    }
}

impl CustomOp for Dinov2UnpackAttnOp {
    fn name(&self) -> &str { "dinov2_unpack_attn" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // input: [BH, N, HD]; output: [B, N, H*HD]
        let bh = input_shapes[0][0];
        let n  = input_shapes[0][1];
        let b  = bh.map(|bh_v| bh_v / self.num_heads);
        vec![b, n, Some(self.num_heads * self.head_dim)]
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
