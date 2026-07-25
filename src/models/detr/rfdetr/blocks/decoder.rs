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


//! DETR transformer decoder.
//!
//! Takes multi-scale feature maps `(s3, s4, s5)` from the MultiScaleProjector
//! and learned object queries, then runs N decoder layers.
//!
//! Multi-scale value preparation pipeline:
//!   s3 [B, C, 2H, 2W] → nchw_to_nld → proj → [B, S3, C]
//!   s4 [B, C,  H,  W] → nchw_to_nld → proj → [B, S4, C]  (S4 = H*W)
//!   s5 [B, C, H/2,W/2]→ nchw_to_nld → proj → [B, S5, C]
//!   cat: [B, S_total, C]  (S_total = S3+S4+S5)
//!   pack_heads: [BH, S_total, HD]
//!
//! Output:
//!   queries [B, Nq, D] — fed to class/box prediction heads

use teeny_core::{
    dtype::Float,
    graph::{CustomData, DtypeRepr, SymTensor},
    name_scope::name_scope,
    nn::{Layer, linear::Linear},
};

use crate::models::detr::rfdetr::kernels::{
    reshape::Dinov2NchwToNldOp,
    seq_ops::{PackHeadsOp, SeqCat2Op},
};

use super::decoder_layer::DecoderLayer;

// ── Value preparation ─────────────────────────────────────────────────────────

/// Project a single `[B, C, H, W]` feature map into `[B, H*W, D]`.
fn project_scale<D: Float + 'static>(
    feat:     SymTensor,          // [B, C, H, W]
    proj:     &Linear<D, SymTensor, SymTensor, 3>,
    embed_dim: i32,
) -> SymTensor {
    // [B, C, H, W] → [B, H*W, C]
    let nld = feat.record_custom(
        CustomData::new(Dinov2NchwToNldOp::new(embed_dim)),
        &[], None,
    );
    // linear projection [B, H*W, C] → [B, H*W, D]
    proj.call(nld)
}

/// Concatenate two `[B, S, D]` tensors along dim 1.
fn seq_cat(a: SymTensor, b: SymTensor, d: i32) -> SymTensor {
    a.record_custom(CustomData::new(SeqCat2Op::new(d)), &[&b], None)
}

// ── Spatial metadata ──────────────────────────────────────────────────────────

/// Build spatial_shapes `[n_levels, 2]` and level_start `[n_levels]` as
/// graph Input nodes (fed as constant buffers at runtime).
///
/// Returns `(spatial_shapes, level_start)` as SymTensors.
fn make_spatial_meta(
    graph: &std::rc::Rc<std::cell::RefCell<teeny_core::graph::Graph>>,
    dtype: DtypeRepr,
    level_shapes: &[(usize, usize)],  // (H_l, W_l) per level
) -> (SymTensor, SymTensor) {
    use teeny_core::graph::Op;
    let n_levels = level_shapes.len();

    let ss_shape = vec![Some(n_levels), Some(2)];
    let ss_id = graph.borrow_mut().add_node(
        Op::Input, vec![], dtype, ss_shape.clone(),
    );
    let spatial_shapes = SymTensor {
        node_id: ss_id,
        graph: graph.clone(),
        dtype,
        shape: ss_shape,
    };

    let ls_shape = vec![Some(n_levels)];
    let ls_id = graph.borrow_mut().add_node(
        Op::Input, vec![], dtype, ls_shape.clone(),
    );
    let level_start = SymTensor {
        node_id: ls_id,
        graph: graph.clone(),
        dtype,
        shape: ls_shape,
    };

    (spatial_shapes, level_start)
}

// ── Decoder ───────────────────────────────────────────────────────────────────

/// DETR transformer decoder.
///
/// Parameters:
/// - Per-scale value projections: `proj_s3`, `proj_s4`, `proj_s5` (each `Linear(C, D)`)
/// - Per-layer: `DecoderLayer` params (self-attn, cross-attn, FFN, layer norms)
///
/// Inputs:
/// - `s3 [B, C, 2H, 2W]`, `s4 [B, C, H, W]`, `s5 [B, C, H/2, W/2]` from projector
/// - `queries [B, Nq, D]` — learnable object queries
///
/// Output: `queries [B, Nq, D]` after N decoder layers
pub struct TransformerDecoder<D: Float> {
    proj_s3:    Linear<D, SymTensor, SymTensor, 3>,
    proj_s4:    Linear<D, SymTensor, SymTensor, 3>,
    proj_s5:    Linear<D, SymTensor, SymTensor, 3>,
    layers:     Vec<DecoderLayer<D>>,
    embed_dim:  usize,
    num_heads:  usize,
    #[allow(dead_code)]
    n_levels:   usize,
    #[allow(dead_code)]
    n_points:   usize,
}

impl<D: Float + 'static> TransformerDecoder<D> {
    /// `embed_dim` — hidden dim (= C = D)
    /// `depth`     — number of decoder layers
    /// `n_points`  — sampling points per head per level in MSDeformAttn (typically 4)
    pub fn new(
        embed_dim: usize,
        depth:     usize,
        num_heads: usize,
        mlp_ratio: usize,
        n_points:  usize,
    ) -> Self {
        let n_levels = 3; // s3, s4, s5
        let mlp_dim  = embed_dim * mlp_ratio;
        Self {
            proj_s3: Linear::new(embed_dim, embed_dim, true),
            proj_s4: Linear::new(embed_dim, embed_dim, true),
            proj_s5: Linear::new(embed_dim, embed_dim, true),
            layers:  (0..depth)
                .map(|_| DecoderLayer::new(embed_dim, num_heads, mlp_dim, n_levels, n_points))
                .collect(),
            embed_dim,
            num_heads,
            n_levels,
            n_points,
        }
    }

    /// Forward pass.
    ///
    /// `s3`, `s4`, `s5` are the three feature scales from `MultiScaleProjector`.
    /// `queries` is the `[B, Nq, D]` learnable query tensor.
    pub fn call(
        &self,
        s3:      SymTensor,
        s4:      SymTensor,
        s5:      SymTensor,
        queries: SymTensor,
    ) -> SymTensor {
        let d     = self.embed_dim as i32;
        let graph = s3.graph.clone();
        let dtype = s3.dtype;

        // ── Level spatial shapes ───────────────────────────────────────────────
        // We compute H/W per scale from the input shapes (static dims known at graph build time).
        let level_shapes: Vec<(usize, usize)> = {
            let h3 = s3.shape[2].unwrap_or(0);
            let w3 = s3.shape[3].unwrap_or(0);
            let h4 = s4.shape[2].unwrap_or(0);
            let w4 = s4.shape[3].unwrap_or(0);
            let h5 = s5.shape[2].unwrap_or(0);
            let w5 = s5.shape[3].unwrap_or(0);
            vec![(h3, w3), (h4, w4), (h5, w5)]
        };
        let (spatial_shapes, level_start) = make_spatial_meta(&graph, dtype, &level_shapes);

        // ── Build multi-scale value [BH, S_total, HD] ─────────────────────────
        let v3 = {
            let _g = name_scope("val_s3");
            project_scale(s3, &self.proj_s3, d)    // [B, S3, D]
        };
        let v4 = {
            let _g = name_scope("val_s4");
            project_scale(s4, &self.proj_s4, d)    // [B, S4, D]
        };
        let v5 = {
            let _g = name_scope("val_s5");
            project_scale(s5, &self.proj_s5, d)    // [B, S5, D]
        };

        // Cat to [B, S_total, D]
        let v34  = seq_cat(v3, v4, d);
        let v345 = seq_cat(v34, v5, d);

        // Pack heads: [B, S_total, D] → [BH, S_total, HD]
        let head_dim = self.embed_dim / self.num_heads;
        let value = v345.record_custom(
            CustomData::new(PackHeadsOp::new(head_dim as i32, self.num_heads)),
            &[], None,
        ); // [BH, S_total, HD]

        // ── Decoder layers ────────────────────────────────────────────────────
        let queries = self.layers.iter().enumerate().fold(queries, |q, (i, layer)| {
            let _g = name_scope(&format!("layer_{i}"));
            layer.call(q, value.clone(), &spatial_shapes, &level_start)
        });

        queries
    }
}
