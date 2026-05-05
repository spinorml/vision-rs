/*
 * SpinorML Ltd 🚀 AGPL-3.0 License - https://spinorml.com/license
 */

//! Simplified TaskAlignedAssigner (CPU-side).
//!
//! For each GT box, alignment scores are:
//!   score = cls_score ^ alpha * iou ^ beta
//!
//! The top-k anchors per GT are selected as positives.  Conflicts (multiple GTs
//! assigning the same anchor) are broken by highest alignment score.

/// Result of assigning GT targets to anchors for a single image.
pub struct AssignResult {
    /// Whether each anchor is a positive. Shape: `[A]`.
    pub is_positive: Vec<bool>,
    /// GT box (cx, cy, w, h) assigned to each anchor (only valid for positives).
    /// Shape: `[4, A]` channels-first.
    pub target_boxes: Vec<f32>,
    /// GT class index assigned to each anchor (only valid for positives).
    /// Shape: `[A]`.
    pub target_cls: Vec<usize>,
    /// Per-anchor IoU with its assigned GT (used as loss weight for positives).
    /// Shape: `[A]`.
    pub iou_weight: Vec<f32>,
}

/// Compute the IoU between a predicted XYWH box and a GT XYWH box.
fn iou_xywh(p: [f32; 4], g: [f32; 4]) -> f32 {
    let (px1, py1) = (p[0] - p[2] * 0.5, p[1] - p[3] * 0.5);
    let (px2, py2) = (p[0] + p[2] * 0.5, p[1] + p[3] * 0.5);
    let (gx1, gy1) = (g[0] - g[2] * 0.5, g[1] - g[3] * 0.5);
    let (gx2, gy2) = (g[0] + g[2] * 0.5, g[1] + g[3] * 0.5);

    let ix1 = px1.max(gx1);
    let iy1 = py1.max(gy1);
    let ix2 = px2.min(gx2);
    let iy2 = py2.min(gy2);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let union = p[2] * p[3] + g[2] * g[3] - inter;
    inter / (union + 1e-7)
}

/// TaskAlignedAssigner parameters.
#[derive(Clone)]
pub struct TaskAlignedAssigner {
    /// Top-k anchors selected per GT.
    pub top_k: usize,
    /// Exponent on the predicted class confidence.
    pub alpha: f32,
    /// Exponent on the predicted IoU.
    pub beta: f32,
}

impl Default for TaskAlignedAssigner {
    fn default() -> Self {
        Self { top_k: 10, alpha: 0.5, beta: 6.0 }
    }
}

impl TaskAlignedAssigner {
    /// Assign GT boxes to anchors for a **single image**.
    ///
    /// # Arguments
    /// * `pred_xywh`   – decoded anchor predictions `[4, A]` channels-first
    /// * `pred_scores` – per-anchor class logits `[nc, A]` channels-first (sigmoid applied internally)
    /// * `anchor_cx`   – anchor centre x `[A]`
    /// * `anchor_cy`   – anchor centre y `[A]`
    /// * `gt_boxes`    – GT boxes `[[cx, cy, w, h]; M]`
    /// * `gt_cls`      – GT class indices `[M]`
    pub fn assign(
        &self,
        pred_xywh: &[f32],
        pred_scores: &[f32],
        anchor_cx: &[f32],
        anchor_cy: &[f32],
        gt_boxes: &[[f32; 4]],
        gt_cls: &[usize],
    ) -> AssignResult {
        let a = anchor_cx.len();
        let m = gt_boxes.len();
        let nc = pred_scores.len() / a;

        let mut target_boxes = vec![0.0f32; 4 * a];
        let mut target_cls   = vec![0usize; a];
        let mut iou_weight   = vec![0.0f32; a];
        let mut assigned_score = vec![-1.0f32; a]; // -1 = unassigned
        let mut is_positive  = vec![false; a];

        if m == 0 {
            return AssignResult { is_positive, target_boxes, target_cls, iou_weight };
        }

        // Pre-compute sigmoid of predicted class scores.
        let sig: Vec<f32> = pred_scores.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();

        for (gi, &gt_box) in gt_boxes.iter().enumerate() {
            let gc = gt_cls[gi];

            // Compute per-anchor alignment scores for this GT.
            let mut scores: Vec<(usize, f32)> = (0..a).filter_map(|ai| {
                // Anchor must be inside the GT box.
                let [gx, gy, gw, gh] = gt_box;
                let cx = anchor_cx[ai];
                let cy = anchor_cy[ai];
                if cx < gx - gw * 0.5 || cx > gx + gw * 0.5
                    || cy < gy - gh * 0.5 || cy > gy + gh * 0.5
                {
                    return None;
                }

                let p = [pred_xywh[ai], pred_xywh[a+ai], pred_xywh[2*a+ai], pred_xywh[3*a+ai]];
                let iou = iou_xywh(p, gt_box).max(0.0);

                // Class score for the GT class.
                let cls_score = if gc < nc { sig[gc * a + ai] } else { 0.0 };

                let score = cls_score.powf(self.alpha) * iou.powf(self.beta);
                Some((ai, score))
            }).collect();

            if scores.is_empty() { continue; }

            // Pick top-k anchors by score.
            scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scores.truncate(self.top_k);

            for (ai, score) in scores {
                if score > assigned_score[ai] {
                    assigned_score[ai] = score;
                    is_positive[ai] = true;
                    target_cls[ai]  = gc;
                    let p = [pred_xywh[ai], pred_xywh[a+ai], pred_xywh[2*a+ai], pred_xywh[3*a+ai]];
                    iou_weight[ai]  = iou_xywh(p, gt_box).max(0.0);
                    target_boxes[ai]         = gt_box[0];
                    target_boxes[a + ai]     = gt_box[1];
                    target_boxes[2 * a + ai] = gt_box[2];
                    target_boxes[3 * a + ai] = gt_box[3];
                }
            }
        }

        AssignResult { is_positive, target_boxes, target_cls, iou_weight }
    }
}
