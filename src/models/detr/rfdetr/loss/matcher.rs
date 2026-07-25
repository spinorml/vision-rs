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


//! Hungarian matching for DETR set prediction loss.
//!
//! Computes an optimal bipartite matching between a set of predicted bounding
//! boxes and a set of ground-truth annotations using the Hungarian (Munkres)
//! algorithm.
//!
//! Cost matrix components (matching only; no gradients needed):
//!   cost_cls  = −p_hat[q, gt_class]          (negative predicted prob, no focal)
//!   cost_bbox = Σ|pred_box[q] − gt_box[j]|   (L1 distance, 4 coords)
//!   cost_giou = −giou(pred_box[q], gt_box[j]) (negative GIoU)
//!
//! The weights λ_cls, λ_bbox, λ_giou are configurable.

/// Bounding box in `[cx, cy, w, h]` normalised coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Box4 {
    pub cx: f32,
    pub cy: f32,
    pub w:  f32,
    pub h:  f32,
}

impl Box4 {
    pub fn from_slice(s: &[f32]) -> Self {
        Self { cx: s[0], cy: s[1], w: s[2], h: s[3] }
    }

    fn x1(&self) -> f32 { self.cx - self.w * 0.5 }
    fn y1(&self) -> f32 { self.cy - self.h * 0.5 }
    fn x2(&self) -> f32 { self.cx + self.w * 0.5 }
    fn y2(&self) -> f32 { self.cy + self.h * 0.5 }
}

// ── GIoU ─────────────────────────────────────────────────────────────────────

/// Compute GIoU between two boxes in `[cx, cy, w, h]` format.
pub fn giou(a: Box4, b: Box4) -> f32 {
    let ax1 = a.x1(); let ay1 = a.y1(); let ax2 = a.x2(); let ay2 = a.y2();
    let bx1 = b.x1(); let by1 = b.y1(); let bx2 = b.x2(); let by2 = b.y2();

    let inter_x1 = ax1.max(bx1);
    let inter_y1 = ay1.max(by1);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter   = inter_w * inter_h;

    let area_a = (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0);
    let area_b = (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0);
    let union   = area_a + area_b - inter;

    let iou = if union > 0.0 { inter / union } else { 0.0 };

    // Enclosing box
    let enc_x1 = ax1.min(bx1);
    let enc_y1 = ay1.min(by1);
    let enc_x2 = ax2.max(bx2);
    let enc_y2 = ay2.max(by2);
    let enc    = (enc_x2 - enc_x1).max(0.0) * (enc_y2 - enc_y1).max(0.0);

    iou - if enc > 0.0 { (enc - union) / enc } else { 0.0 }
}

// ── Hungarian matcher ─────────────────────────────────────────────────────────

/// Weights for the three cost components.
#[derive(Clone, Copy, Debug)]
pub struct MatchWeights {
    pub class: f32,
    pub bbox:  f32,
    pub giou:  f32,
}

impl Default for MatchWeights {
    fn default() -> Self { Self { class: 1.0, bbox: 5.0, giou: 2.0 } }
}

/// Matched pair `(query_idx, gt_idx)`.
pub type Match = (usize, usize);

/// Hungarian matcher: finds the minimum-cost assignment between `n_queries`
/// predictions and `n_gt` ground-truth instances.
///
/// - `pred_logits`: `[n_queries, n_classes]` raw logits (after sigmoid → prob)
/// - `pred_boxes`:  `[n_queries, 4]` in cx/cy/w/h normalised
/// - `gt_classes`:  `[n_gt]` integer class indices
/// - `gt_boxes`:    `[n_gt, 4]` in cx/cy/w/h normalised
///
/// Returns at most `n_gt` matched pairs (one per ground-truth instance).
pub fn hungarian_match(
    pred_logits: &[f32],   // [Nq, C]
    pred_boxes:  &[f32],   // [Nq, 4]
    gt_classes:  &[usize], // [Ng]
    gt_boxes:    &[f32],   // [Ng, 4]
    n_queries:   usize,
    n_classes:   usize,
    weights:     MatchWeights,
) -> Vec<Match> {
    let n_gt = gt_classes.len();
    if n_gt == 0 { return Vec::new(); }

    // Build cost matrix [n_gt, n_queries] (Hungarian finds row→col assignment)
    let mut cost = vec![0.0f32; n_gt * n_queries];

    for j in 0..n_gt {
        let gt_cls = gt_classes[j];
        let gb = Box4::from_slice(&gt_boxes[j * 4..]);

        for i in 0..n_queries {
            // Sigmoid of the predicted logit for the GT class
            let logit = pred_logits[i * n_classes + gt_cls];
            let prob  = 1.0 / (1.0 + (-logit).exp());

            // Classification cost
            let c_cls  = -prob;

            // L1 box cost
            let pb = Box4::from_slice(&pred_boxes[i * 4..]);
            let c_bbox = (pb.cx - gb.cx).abs()
                       + (pb.cy - gb.cy).abs()
                       + (pb.w  - gb.w ).abs()
                       + (pb.h  - gb.h ).abs();

            // GIoU cost
            let c_giou = -giou(pb, gb);

            cost[j * n_queries + i] =
                weights.class * c_cls
                + weights.bbox  * c_bbox
                + weights.giou  * c_giou;
        }
    }

    // Run the Hungarian algorithm on the [n_gt × n_queries] cost matrix
    hungarian_algorithm(&cost, n_gt, n_queries)
}

// ── Hungarian algorithm (Kuhn-Munkres, O(n³)) ────────────────────────────────

fn hungarian_algorithm(cost: &[f32], n_rows: usize, n_cols: usize) -> Vec<Match> {
    // Pad to square by augmenting with large costs
    let n = n_rows.max(n_cols);
    let inf = f32::MAX / 2.0;

    let mut padded = vec![inf; n * n];
    for r in 0..n_rows {
        for c in 0..n_cols {
            padded[r * n + c] = cost[r * n_cols + c];
        }
    }

    // Row reduction
    for r in 0..n {
        let min_val = padded[r * n..r * n + n].iter().cloned().fold(f32::MAX, f32::min);
        for c in 0..n { padded[r * n + c] -= min_val; }
    }

    // Column reduction
    for c in 0..n {
        let min_val = (0..n).map(|r| padded[r * n + c]).fold(f32::MAX, f32::min);
        for r in 0..n { padded[r * n + c] -= min_val; }
    }

    let mut row_of_col = vec![usize::MAX; n]; // assignment: col → row
    let mut col_of_row = vec![usize::MAX; n]; // assignment: row → col

    // Try to find augmenting paths until we have a full matching
    for row in 0..n {
        // BFS/DFS for augmenting path starting from `row`
        let mut visited_cols = vec![false; n];
        augment(row, &padded, n, &mut row_of_col, &mut col_of_row, &mut visited_cols);
    }

    // Collect valid matches (only for original non-padded rows/cols)
    (0..n_rows)
        .filter_map(|r| {
            let c = col_of_row[r];
            if c < n_cols { Some((c, r)) } // (query_idx=c, gt_idx=r)
            else { None }
        })
        .collect()
}

fn augment(
    row: usize,
    cost: &[f32],
    n: usize,
    row_of_col: &mut Vec<usize>,
    col_of_row: &mut Vec<usize>,
    visited: &mut Vec<bool>,
) -> bool {
    let eps = 1e-6f32;
    for col in 0..n {
        if !visited[col] && cost[row * n + col].abs() < eps {
            visited[col] = true;
            if row_of_col[col] == usize::MAX
                || augment(row_of_col[col], cost, n, row_of_col, col_of_row, visited)
            {
                row_of_col[col] = row;
                col_of_row[row] = col;
                return true;
            }
        }
    }
    false
}
