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


//! DETR decoder layer: self-attention + MSDeformAttn cross-attention + FFN.
//!
//! One decoder layer:
//!   LayerNorm(q) → self_attn(q, q, q) → Add residual q
//!   LayerNorm(q) → ms_deform_cross_attn(q, value) → Add residual q
//!   LayerNorm(q) → FFN(q) → Add residual q
//!
//! Input/output queries: `[B, Nq, D]`

use teeny_core::{
    dtype::Float,
    graph::{CustomData, Op, SymTensor},
    name_scope::name_scope,
    nn::{
        Layer,
        activation::relu::Relu,
        layernorm::LayerNorm,
        linear::Linear,
    },
};

use crate::models::detr::rfdetr::kernels::{
    mha::{Dinov2PackQkvOp, Dinov2UnpackAttnOp, FlashAttn2Dinov2Op},
    ms_deform_attn::MsDeformAttnOp,
    seq_ops::PackHeadsOp,
};

// ── Graph helpers ─────────────────────────────────────────────────────────────

fn add(a: SymTensor, b: SymTensor) -> SymTensor {
    let shape = a.shape.clone();
    let node_id = a.graph.borrow_mut().add_node(
        Op::Add, vec![a.node_id, b.node_id], a.dtype, shape.clone(),
    );
    SymTensor { node_id, graph: a.graph.clone(), dtype: a.dtype, shape }
}

// ── Cross-attention via MSDeformAttn ──────────────────────────────────────────

/// MSDeformable cross-attention.
///
/// Inputs:
/// - `queries [B, Nq, D]` — query hidden states
/// - `value [BH, S_total, HD]` — pre-packed multi-scale value features
/// - `spatial_shapes [n_levels, 2]` — (Hl, Wl) per scale as f32
/// - `level_start [n_levels]` — cumulative token offset per scale as f32
///
/// Output: `[B, Nq, D]`
struct MsDeformCrossAttn<D: Float> {
    sampling_locs: Linear<D, SymTensor, SymTensor, 3>, // [B, Nq, D] → [B, Nq, H*L*P*2]
    attn_weights:  Linear<D, SymTensor, SymTensor, 3>, // [B, Nq, D] → [B, Nq, H*L*P]
    value_proj:    Linear<D, SymTensor, SymTensor, 3>, // placeholder for output proj
    head_dim:      usize,
    n_heads:       usize,
    n_levels:      usize,
    n_points:      usize,
}

impl<D: Float + 'static> MsDeformCrossAttn<D> {
    fn new(embed_dim: usize, n_heads: usize, n_levels: usize, n_points: usize) -> Self {
        let head_dim = embed_dim / n_heads;
        let locs_out  = n_heads * n_levels * n_points * 2;
        let weigh_out = n_heads * n_levels * n_points;
        Self {
            sampling_locs: Linear::new(embed_dim, locs_out, true),
            attn_weights:  Linear::new(embed_dim, weigh_out, true),
            value_proj:    Linear::new(embed_dim, embed_dim, true),
            head_dim,
            n_heads,
            n_levels,
            n_points,
        }
    }

    fn call(
        &self,
        queries: SymTensor,             // [B, Nq, D]
        value:   SymTensor,             // [BH, S_total, HD]
        spatial_shapes: &SymTensor,     // [n_levels, 2]
        level_start:    &SymTensor,     // [n_levels]
    ) -> SymTensor {
        // sampling locations: [B, Nq, D] → [B, Nq, H*L*P*2] → sigmoid → [BH, Nq, L*P*2]
        let locs = self.sampling_locs.call(queries.clone()); // [B, Nq, H*L*P*2]
        let locs = {
            let shape = locs.shape.clone();
            let node_id = locs.graph.borrow_mut().add_node(
                Op::Sigmoid, vec![locs.node_id], locs.dtype, shape.clone(),
            );
            SymTensor { node_id, graph: locs.graph.clone(), dtype: locs.dtype, shape }
        };
        let locs = locs.record_custom(
            CustomData::new(PackHeadsOp::new(
                (self.n_levels * self.n_points * 2) as i32,
                self.n_heads,
            )),
            &[],
            None,
        ); // [BH, Nq, L*P*2]

        // attention weights: [B, Nq, D] → [B, Nq, H*L*P] → softmax → [BH, Nq, L*P]
        let weights = self.attn_weights.call(queries.clone()); // [B, Nq, H*L*P]
        let weights = {
            let shape = weights.shape.clone();
            let node_id = weights.graph.borrow_mut().add_node(
                Op::Softmax { dim: 2 }, vec![weights.node_id], weights.dtype, shape.clone(),
            );
            SymTensor { node_id, graph: weights.graph.clone(), dtype: weights.dtype, shape }
        };
        let weights = weights.record_custom(
            CustomData::new(PackHeadsOp::new(
                (self.n_levels * self.n_points) as i32,
                self.n_heads,
            )),
            &[],
            None,
        ); // [BH, Nq, L*P]

        // MSDeformAttn: value [BH, S_total, HD], locs [BH, Nq, L*P*2], weights [BH, Nq, L*P]
        //               spatial_shapes [n_levels, 2], level_start [n_levels]
        // Output: [BH, Nq, HD]
        let attn_out = value.record_custom(
            CustomData::new(MsDeformAttnOp::new(
                self.head_dim as i32,
                self.n_levels,
                self.n_points,
            )),
            &[&locs, &weights, spatial_shapes, level_start],
            None,
        ); // [BH, Nq, HD]

        // Unpack heads: [BH, Nq, HD] → [B, Nq, D]
        let out = attn_out.record_custom(
            CustomData::new(Dinov2UnpackAttnOp::new(self.head_dim as i32, self.n_heads)),
            &[],
            None,
        );

        // Output projection
        self.value_proj.call(out)
    }
}

// ── Decoder layer ─────────────────────────────────────────────────────────────

/// One DETR transformer decoder layer.
///
/// Parameters:
/// - ln1/ln2/ln3: LayerNorm weight+bias `[D]`
/// - qkv/proj (self-attn), sampling_locs/attn_weights/value_proj (cross-attn)
/// - fc1 `[mlp_dim, D]`, fc2 `[D, mlp_dim]`
///
/// Inputs:
/// - `queries [B, Nq, D]`
/// - `value [BH, S_total, HD]` — multi-scale value (packed heads)
/// - `spatial_shapes [n_levels, 2]`
/// - `level_start [n_levels]`
pub struct DecoderLayer<D: Float> {
    ln1:         LayerNorm<D, SymTensor, SymTensor, 3>,
    qkv:         Linear<D, SymTensor, SymTensor, 3>,
    proj:        Linear<D, SymTensor, SymTensor, 3>,
    ln2:         LayerNorm<D, SymTensor, SymTensor, 3>,
    cross_attn:  MsDeformCrossAttn<D>,
    ln3:         LayerNorm<D, SymTensor, SymTensor, 3>,
    fc1:         Linear<D, SymTensor, SymTensor, 3>,
    fc2:         Linear<D, SymTensor, SymTensor, 3>,
    relu:        Relu<D, SymTensor, 3>,
    embed_dim:   usize,
    num_heads:   usize,
}

impl<D: Float + 'static> DecoderLayer<D> {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        mlp_dim:   usize,
        n_levels:  usize,
        n_points:  usize,
    ) -> Self {
        let _head_dim = embed_dim / num_heads;
        Self {
            ln1:        LayerNorm::new([embed_dim]),
            qkv:        Linear::new(embed_dim, 3 * embed_dim, true),
            proj:       Linear::new(embed_dim, embed_dim, true),
            ln2:        LayerNorm::new([embed_dim]),
            cross_attn: MsDeformCrossAttn::new(embed_dim, num_heads, n_levels, n_points),
            ln3:        LayerNorm::new([embed_dim]),
            fc1:        Linear::new(embed_dim, mlp_dim, true),
            fc2:        Linear::new(mlp_dim, embed_dim, true),
            relu:       Relu::new(),
            embed_dim,
            num_heads,
        }
    }

    pub fn call(
        &self,
        queries:        SymTensor,
        value:          SymTensor,
        spatial_shapes: &SymTensor,
        level_start:    &SymTensor,
    ) -> SymTensor {
        let head_dim = self.embed_dim / self.num_heads;
        let residual = queries.clone();

        // ── Self-attention ────────────────────────────────────────────────────
        let q = { let _g = name_scope("ln1"); self.ln1.call(queries) };

        let qkv_out = { let _g = name_scope("qkv"); self.qkv.call(q) };

        let packed = qkv_out.record_custom(
            CustomData::new(Dinov2PackQkvOp::new(head_dim as i32, self.num_heads)),
            &[], None,
        );
        let attn_out = packed.record_custom(
            CustomData::new(FlashAttn2Dinov2Op::new(head_dim as i32)),
            &[], None,
        );
        let unpacked = attn_out.record_custom(
            CustomData::new(Dinov2UnpackAttnOp::new(head_dim as i32, self.num_heads)),
            &[], None,
        );
        let self_attn_out = { let _g = name_scope("proj"); self.proj.call(unpacked) };
        let q = add(residual, self_attn_out);

        // ── Cross-attention ───────────────────────────────────────────────────
        let residual = q.clone();
        let q_ln = { let _g = name_scope("ln2"); self.ln2.call(q) };

        let cross_out = {
            let _g = name_scope("cross_attn");
            self.cross_attn.call(q_ln, value, spatial_shapes, level_start)
        };
        let q = add(residual, cross_out);

        // ── FFN ───────────────────────────────────────────────────────────────
        let residual = q.clone();
        let q = { let _g = name_scope("ln3"); self.ln3.call(q) };
        let q = { let _g = name_scope("fc1"); self.fc1.call(q) };
        let q = self.relu.call(q);
        let q = { let _g = name_scope("fc2"); self.fc2.call(q) };

        add(residual, q)
    }
}
