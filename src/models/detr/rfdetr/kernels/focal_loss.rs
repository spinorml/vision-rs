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


//! Sigmoid focal loss Triton kernel for RF-DETR classification training.
//!
//! Computes the focal loss from Lin et al. (2017):
//!   `p = sigmoid(x)`
//!   `p_t = p * t + (1 - p) * (1 - t)`    (correct-class probability)
//!   `alpha_t = alpha * t + (1 - alpha) * (1 - t)`
//!   `ce = -log(p_t)`                       (binary cross-entropy)
//!   `loss[i] = alpha_t * (1 - p_t)^gamma * ce / num_boxes`
//!
//! Layout (flat 1-D, N = total elements = num_queries * num_classes):
//!   `logits`:  `[N]` — predicted class logits
//!   `targets`: `[N]` — binary targets ∈ {0, 1}
//!   `loss`:    `[N]` — per-element focal loss (sum externally if needed)
//!
//! Grid: `cdiv(N, BLOCK_N)` flat CTAs.

#![allow(non_snake_case)]

use teeny_macros::kernel;
use teeny_triton::triton::{
    types::{AddOffsets, Comparison},
    *,
};

// ── sigmoid_focal_loss_forward ────────────────────────────────────────────────

/// Forward kernel: logits + targets → per-element focal loss.
///
/// Grid: `cdiv(N, BLOCK_N)`.
#[allow(clippy::erasing_op, clippy::identity_op)]
#[kernel]
pub fn sigmoid_focal_loss_forward<T: Triton, const BLOCK_N: i32>(
    logits_ptr: T::Pointer<f32>,  // [N]
    targets_ptr: T::Pointer<f32>, // [N]
    loss_ptr: T::Pointer<f32>,    // [N]
    N: i32,
    alpha: f32,
    gamma: f32,
    num_boxes: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let n_start = T::program_id(Axis::X) * BLOCK_N;
    let n_offs  = T::arange(0, BLOCK_N) + n_start;
    let mask    = n_offs.lt(N);

    let zeros = T::zeros::<f32>(&[BLOCK_N]);
    let ones  = T::full::<f32>(&[BLOCK_N], 1.0_f32);

    let x = T::load(logits_ptr.add_offsets(n_offs),  Some(mask), Some(zeros), &[], None, None, None, false);
    let t = T::load(targets_ptr.add_offsets(n_offs), Some(mask), Some(zeros), &[], None, None, None, false);

    // Numerically stable sigmoid: sigma(x) = 1 / (1 + exp(-x))
    let p = T::sigmoid(x);

    // Correct-class probability
    let p_t = p * t + (ones - p) * (ones - t);

    // Alpha weight
    let alpha_t_val = T::full::<f32>(&[BLOCK_N], alpha);
    let one_minus_a = T::full::<f32>(&[BLOCK_N], 1.0_f32 - alpha);
    let alpha_t     = alpha_t_val * t + one_minus_a * (ones - t);

    // Focal weight: (1 - p_t)^gamma
    let one_minus_pt = ones - p_t;
    let focal_weight = T::exp(T::full::<f32>(&[BLOCK_N], gamma) * T::log(one_minus_pt + T::full::<f32>(&[BLOCK_N], 1e-8)));

    // Binary cross-entropy: -log(p_t + eps)
    let ce = zeros - T::log(p_t + T::full::<f32>(&[BLOCK_N], 1e-8));

    // Focal loss normalised by num_boxes
    let loss = alpha_t * focal_weight * ce / T::full::<f32>(&[BLOCK_N], num_boxes);

    T::store(loss_ptr.add_offsets(n_offs), loss, Some(mask), &[], None, None);
}

// ── sigmoid_focal_loss_backward ───────────────────────────────────────────────

/// Backward kernel: d_loss (upstream) + logits + targets → d_logits.
///
/// Closed-form gradient:
///   Let `p = sigmoid(x)`, `p_t = p*t + (1-p)*(1-t)`.
///   `d_loss/d_x = alpha_t / num_boxes *
///                 [(1-p_t)^gamma * (sigma(x)-t)
///                  - gamma * (1-p_t)^(gamma-1) * (-log(p_t)) * (2t-1) * p*(1-p)]`
///
/// Grid: `cdiv(N, BLOCK_N)`.
#[allow(clippy::erasing_op, clippy::identity_op)]
#[kernel]
pub fn sigmoid_focal_loss_backward<T: Triton, const BLOCK_N: i32>(
    logits_ptr: T::Pointer<f32>,   // [N] forward input
    targets_ptr: T::Pointer<f32>,  // [N] forward input
    d_loss_ptr: T::Pointer<f32>,   // [N] upstream gradient
    d_logits_ptr: T::Pointer<f32>, // [N] output gradient
    N: i32,
    alpha: f32,
    gamma: f32,
    num_boxes: f32,
) where
    T::I32Tensor: types::Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let n_start = T::program_id(Axis::X) * BLOCK_N;
    let n_offs  = T::arange(0, BLOCK_N) + n_start;
    let mask    = n_offs.lt(N);

    let zeros  = T::zeros::<f32>(&[BLOCK_N]);
    let ones   = T::full::<f32>(&[BLOCK_N], 1.0_f32);
    let eps    = T::full::<f32>(&[BLOCK_N], 1e-8_f32);
    let gamma_t = T::full::<f32>(&[BLOCK_N], gamma);
    let two    = T::full::<f32>(&[BLOCK_N], 2.0_f32);

    let x    = T::load(logits_ptr.add_offsets(n_offs),  Some(mask), Some(zeros), &[], None, None, None, false);
    let t    = T::load(targets_ptr.add_offsets(n_offs), Some(mask), Some(zeros), &[], None, None, None, false);
    let dy   = T::load(d_loss_ptr.add_offsets(n_offs),  Some(mask), Some(zeros), &[], None, None, None, false);

    let p   = T::sigmoid(x);
    let p_t = p * t + (ones - p) * (ones - t);

    let alpha_t   = T::full::<f32>(&[BLOCK_N], alpha) * t
                  + T::full::<f32>(&[BLOCK_N], 1.0_f32 - alpha) * (ones - t);
    let one_m_pt  = ones - p_t;
    let log_pt    = T::log(p_t + eps);
    let focal_pow = T::exp(gamma_t * T::log(one_m_pt + eps));

    // d(ce)/d(x) = sigma(x) - t
    let d_ce = p - t;

    // d(focal_weight)/d(x) via chain rule:
    //   d((1-p_t)^gamma)/d(x) = gamma*(1-p_t)^(gamma-1) * (-d(p_t)/d(x))
    //   d(p_t)/d(x) = (2t-1) * p*(1-p)
    let d_pt_dx     = (two * t - ones) * p * (ones - p);
    let d_focal_dx  = (zeros - gamma_t) * T::exp((gamma_t - ones) * T::log(one_m_pt + eps)) * d_pt_dx;

    // d(loss)/d(x) = alpha_t/num_boxes * (focal * d_ce + ce * d_focal)
    let nb   = T::full::<f32>(&[BLOCK_N], num_boxes);
    let ce   = zeros - log_pt;
    let d_logits = dy * alpha_t / nb * (focal_pow * d_ce + ce * d_focal_dx);

    T::store(d_logits_ptr.add_offsets(n_offs), d_logits, Some(mask), &[], None, None);
}

// ── RuntimeOp ─────────────────────────────────────────────────────────────────

pub struct SigmoidFocalLossRuntimeOp {
    fwd: SigmoidFocalLossForward,
    bwd: SigmoidFocalLossBackward,
    alpha: f32,
    gamma: f32,
    num_boxes: f32,
}

impl SigmoidFocalLossRuntimeOp {
    pub fn new(block_n: i32, alpha: f32, gamma: f32, num_boxes: f32) -> Self {
        Self {
            fwd: SigmoidFocalLossForward::new(block_n),
            bwd: SigmoidFocalLossBackward::new(block_n),
            alpha,
            gamma,
            num_boxes,
        }
    }

    pub fn kernel_name(&self) -> &str { self.fwd.name }
    pub fn forward_source(&self) -> &str { &self.fwd.source }
    pub fn backward_source(&self) -> &str { &self.bwd.source }
}

impl teeny_core::model::RuntimeOp for SigmoidFocalLossRuntimeOp {
    fn n_activation_inputs(&self) -> usize { 1 } // gradient only for logits

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
        // inputs: [logits [N], targets [N]]
        let n = inputs[0].1.iter().product::<usize>() as i32;
        visitor.visit_ptr(inputs[0].0);  // logits_ptr
        visitor.visit_ptr(inputs[1].0);  // targets_ptr
        visitor.visit_ptr(output);       // loss_ptr
        visitor.visit_i32(n);
        visitor.visit_f32(self.alpha);
        visitor.visit_f32(self.gamma);
        visitor.visit_f32(self.num_boxes);
    }

    fn block(&self) -> [u32; 3] { [1, 1, 1] } // grid has one thread per CTA; actual tile is BLOCK_N

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        let block_n = 128usize; // must match const generic BLOCK_N
        [n.div_ceil(block_n) as u32, 1, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let n = output_shape.iter().product::<usize>() as i32;
        visitor.visit_ptr(inputs[0].0);  // logits_ptr
        visitor.visit_ptr(inputs[1].0);  // targets_ptr
        visitor.visit_ptr(grad_output);  // d_loss_ptr
        visitor.visit_ptr(grad_inputs[0]); // d_logits_ptr
        visitor.visit_i32(n);
        visitor.visit_f32(self.alpha);
        visitor.visit_f32(self.gamma);
        visitor.visit_f32(self.num_boxes);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [1, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let n: usize = output_shape.iter().product();
        let block_n = 128usize;
        [n.div_ceil(block_n) as u32, 1, 1]
    }
}

// ── CustomOp ──────────────────────────────────────────────────────────────────

use std::any::Any;
use std::sync::Arc;
use teeny_core::{
    graph::{CustomOp, Shape},
    model::RuntimeOp,
};

/// Graph node: `[logits [N], targets [N]]` → `loss [N]`
pub struct SigmoidFocalLossOp {
    inner: Arc<SigmoidFocalLossRuntimeOp>,
}

impl SigmoidFocalLossOp {
    /// `block_n`: tile width (should match const generic, typical = 128).
    pub fn new(block_n: i32, alpha: f32, gamma: f32, num_boxes: f32) -> Self {
        Self { inner: Arc::new(SigmoidFocalLossRuntimeOp::new(block_n, alpha, gamma, num_boxes)) }
    }
}

impl CustomOp for SigmoidFocalLossOp {
    fn name(&self) -> &str { "sigmoid_focal_loss_forward" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].to_vec() // same shape as logits
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
