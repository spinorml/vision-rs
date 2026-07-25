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


//! RF-DETR model builder.
//!
//! Combines DINOv2 ViT backbone → MultiScaleProjector → TransformerDecoder
//! → class/box prediction heads into a single forward function.
//!
//! The compiled graph has **two input slots**:
//!   - `input[0]`: image  `[B, 3, img_h, img_w]`
//!   - `input[1]`: queries `[B, Nq, neck_dim]`  (learnable; created as an `Op::Input` inside
//!                                                `rfdetr()` so callers pass only the image)
//!
//! # Graph outputs:
//!   - `class_logits`: `[B, Nq, nc]`  — raw class logits (no activation)
//!   - `box_preds`:    `[B, Nq, 4]`   — normalised cx/cy/w/h (sigmoid applied)

#![allow(clippy::module_inception)]

use teeny_core::{
    dtype::Float,
    graph::{Op, SymTensor},
    nn::{Layer, activation::sigmoid::Sigmoid, linear::Linear},
};

use crate::models::detr::rfdetr::blocks::{
    decoder::TransformerDecoder,
    dinov2_backbone::dinov2_backbone,
    multi_scale_proj::MultiScaleProjector,
};

// ── Variant ───────────────────────────────────────────────────────────────────

/// RF-DETR model variant (selects DINOv2 backbone size).
#[derive(Clone, Copy, Debug)]
pub enum RfDetrVariant {
    /// DINOv2-Small backbone (embed=384, heads=6).
    S,
    /// DINOv2-Base backbone (embed=768, heads=12).
    B,
}

struct RfDetrConfig {
    backbone_dim:   usize, // ViT embed_dim
    backbone_depth: usize, // ViT depth
    backbone_heads: usize, // ViT num_heads
    patch_size:     usize, // typically 14
    neck_dim:       usize, // projector + decoder dim (always 256)
    decoder_depth:  usize,
    decoder_heads:  usize,
    n_points:       usize, // deformable attn sampling points per head per level
    n_queries:      usize, // object queries
}

impl RfDetrVariant {
    fn config(self) -> RfDetrConfig {
        match self {
            RfDetrVariant::S => RfDetrConfig {
                backbone_dim:   384,
                backbone_depth: 12,
                backbone_heads: 6,
                patch_size:     14,
                neck_dim:       256,
                decoder_depth:  6,
                decoder_heads:  8,
                n_points:       4,
                n_queries:      300,
            },
            RfDetrVariant::B => RfDetrConfig {
                backbone_dim:   768,
                backbone_depth: 12,
                backbone_heads: 12,
                patch_size:     14,
                neck_dim:       256,
                decoder_depth:  6,
                decoder_heads:  8,
                n_points:       4,
                n_queries:      300,
            },
        }
    }

    /// Number of object queries.
    pub fn n_queries(self) -> usize { self.config().n_queries }

    /// Decoder and projector channel width.
    pub fn neck_dim(self) -> usize { self.config().neck_dim }
}

// ── Model builder ─────────────────────────────────────────────────────────────

/// Build an RF-DETR model graph.
///
/// Returns a closure that accepts the image `SymTensor`:
/// - `img`: `[B, 3, img_h, img_w]`
///
/// and returns `(class_logits [B, Nq, nc], box_preds [B, Nq, 4])`.
///
/// The object queries are created as a second `Op::Input` node inside the closure,
/// making the compiled model expect two inputs: `input[0]` = image, `input[1]` = queries.
/// Pass `queries [B, Nq, neck_dim]` as the second `TensorRef` in `forward_train`.
///
/// `box_preds` have sigmoid applied → normalised `cx/cy/w/h ∈ [0, 1]`.
pub fn rfdetr<D: Float + 'static>(
    nc:      usize,
    variant: RfDetrVariant,
    img_h:   usize,
    img_w:   usize,
) -> impl Fn(SymTensor) -> (SymTensor, SymTensor) {
    let cfg = variant.config();

    let backbone   = dinov2_backbone::<D>(
        cfg.backbone_dim,
        cfg.backbone_depth,
        cfg.backbone_heads,
        4,              // mlp_ratio — standard for DINOv2
        cfg.patch_size,
        img_h,
        img_w,
    );
    let projector  = MultiScaleProjector::<D>::new(cfg.backbone_dim, cfg.neck_dim, 1);
    let decoder    = TransformerDecoder::<D>::new(
        cfg.neck_dim,
        cfg.decoder_depth,
        cfg.decoder_heads,
        4,              // mlp_ratio for decoder FFN
        cfg.n_points,
    );
    let class_head: Linear<D, SymTensor, SymTensor, 3> = Linear::new(cfg.neck_dim, nc,  true);
    let box_head:   Linear<D, SymTensor, SymTensor, 3> = Linear::new(cfg.neck_dim, 4,   true);
    let box_sigmoid: Sigmoid<D, SymTensor, 3>          = Sigmoid::new();

    move |img: SymTensor| {
        // ── Object queries (second graph input) ───────────────────────────────
        // Created here so callers only pass the image to the builder closure.
        // The compiled model will have input[0]=image, input[1]=queries.
        let queries = {
            let batch = img.shape[0];
            let shape = vec![batch, Some(cfg.n_queries), Some(cfg.neck_dim)];
            let node_id = img.graph.borrow_mut().add_node(
                Op::Input, vec![], img.dtype, shape.clone(),
            );
            SymTensor { node_id, graph: img.graph.clone(), dtype: img.dtype, shape }
        };

        // ── Backbone ──────────────────────────────────────────────────────────
        // [B, 3, img_h, img_w] → [B, backbone_dim, H_p, W_p]
        let feats = backbone(img);

        // ── Multi-scale projector ─────────────────────────────────────────────
        // [B, backbone_dim, H_p, W_p] → (s3, s4, s5)
        let (s3, s4, s5) = projector.call(feats);

        // ── Transformer decoder ───────────────────────────────────────────────
        // s3/s4/s5 + queries [B, Nq, D] → [B, Nq, D]
        let out = decoder.call(s3, s4, s5, queries);

        // ── Prediction heads ──────────────────────────────────────────────────
        let class_logits = class_head.call(out.clone());            // [B, Nq, nc]
        let box_preds    = box_sigmoid.call(box_head.call(out));    // [B, Nq, 4] ∈ (0,1)

        (class_logits, box_preds)
    }
}
