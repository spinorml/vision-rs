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


//! RF-DETR set-matching criterion.
//!
//! `SetCriterion` combines:
//!   - Hungarian matching (no gradient)
//!   - Sigmoid focal loss on matched class predictions
//!   - L1 + GIoU loss on matched box predictions
//!
//! The per-image loss is normalised by the average number of ground-truth
//! boxes across the batch (as in the original DETR paper).

use teeny_core::{
    dtype::Float,
    graph::{CustomData, SymTensor},
    nn::linear::Linear,
};

use crate::models::detr::rfdetr::kernels::focal_loss::SigmoidFocalLossOp;

use super::matcher::{MatchWeights, hungarian_match};

// ── Prediction heads ──────────────────────────────────────────────────────────

/// Class + box prediction heads applied to decoder output queries.
///
/// Input:  `[B, Nq, D]`
/// Outputs: `class_logits [B, Nq, C]`, `box_preds [B, Nq, 4]`
pub struct PredictionHeads<D: Float> {
    class_head: Linear<D, SymTensor, SymTensor, 3>,
    box_head:   Linear<D, SymTensor, SymTensor, 3>,
}

impl<D: Float + 'static> PredictionHeads<D> {
    pub fn new(embed_dim: usize, num_classes: usize) -> Self {
        Self {
            class_head: Linear::new(embed_dim, num_classes, true),
            box_head:   Linear::new(embed_dim, 4, true),
        }
    }

    /// Returns `(class_logits [B, Nq, C], box_preds [B, Nq, 4])`.
    pub fn call(&self, queries: SymTensor) -> (SymTensor, SymTensor) {
        use teeny_core::nn::Layer;
        let cls  = self.class_head.call(queries.clone());
        let bbox = self.box_head.call(queries);
        (cls, bbox)
    }
}

// ── SetCriterion config ───────────────────────────────────────────────────────

/// Loss weights for `SetCriterion`.
#[derive(Clone, Copy, Debug)]
pub struct LossWeights {
    pub cls:  f32,
    pub bbox: f32,
    pub giou: f32,
}

impl Default for LossWeights {
    fn default() -> Self { Self { cls: 1.0, bbox: 5.0, giou: 2.0 } }
}

// ── Graph-side loss functions ─────────────────────────────────────────────────

/// Sigmoid focal loss graph node for matched predictions.
///
/// `logits [N, C]`, `targets [N, C]` (one-hot encoded for matched GTs).
/// Returns per-element losses `[N, C]`; caller sums/normalises.
pub fn focal_loss_graph(
    logits:    SymTensor,
    targets:   SymTensor,
    alpha:     f32,
    gamma:     f32,
    num_boxes: f32,
) -> SymTensor {
    logits.record_custom(
        CustomData::new(SigmoidFocalLossOp::new(128, alpha, gamma, num_boxes)),
        &[&targets],
        None,
    )
}

// ── L1 and GIoU helpers (CPU, matching only) ──────────────────────────────────

/// Compute L1 distance between two 4-element box slices (cx/cy/w/h).
pub fn l1_box(a: &[f32], b: &[f32]) -> f32 {
    (0..4).map(|i| (a[i] - b[i]).abs()).sum()
}

/// Compute GIoU between two box slices.
pub fn giou_box(a: &[f32], b: &[f32]) -> f32 {
    super::matcher::giou(
        super::matcher::Box4::from_slice(a),
        super::matcher::Box4::from_slice(b),
    )
}

// ── SetCriterion ──────────────────────────────────────────────────────────────

/// DETR set-matching loss criterion.
///
/// Usage during training:
/// 1. Run forward pass → `(class_logits [B, Nq, C], box_preds [B, Nq, 4])`
/// 2. Call `match_batch` to get per-image assignments (no gradient)
/// 3. Call `compute_loss_graph` to get differentiable loss nodes
pub struct SetCriterion {
    pub num_classes: usize,
    pub alpha:       f32,
    pub gamma:       f32,
    pub match_w:     MatchWeights,
    pub loss_w:      LossWeights,
}

impl SetCriterion {
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            alpha:   0.25,
            gamma:   2.0,
            match_w: MatchWeights::default(),
            loss_w:  LossWeights::default(),
        }
    }

    /// Run Hungarian matching for a single image.
    ///
    /// - `logits_img [Nq, C]`    — class logits (flat, row-major)
    /// - `boxes_img  [Nq, 4]`    — box predictions (cx/cy/w/h)
    /// - `gt_classes [Ng]`       — GT class labels
    /// - `gt_boxes   [Ng, 4]`    — GT boxes (cx/cy/w/h)
    ///
    /// Returns matched `(query_idx, gt_idx)` pairs.
    pub fn match_image(
        &self,
        logits_img: &[f32],
        boxes_img:  &[f32],
        gt_classes: &[usize],
        gt_boxes:   &[f32],
        n_queries:  usize,
    ) -> Vec<(usize, usize)> {
        hungarian_match(
            logits_img,
            boxes_img,
            gt_classes,
            gt_boxes,
            n_queries,
            self.num_classes,
            self.match_w,
        )
    }
}
