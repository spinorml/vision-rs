// Hungarian matcher tests.
//
// Verifies that the matcher correctly finds optimal bipartite assignments
// on small hand-crafted examples.

use vision_rs::models::detr::rfdetr::loss::matcher::{
    giou, Box4, MatchWeights, hungarian_match,
};

// ── GIoU correctness ──────────────────────────────────────────────────────────

#[test]
fn test_giou_identical_boxes() {
    let b = Box4 { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 };
    let g = giou(b, b);
    assert!((g - 1.0).abs() < 1e-5, "identical boxes → GIoU = 1.0, got {g}");
}

#[test]
fn test_giou_no_overlap() {
    let a = Box4 { cx: 0.1, cy: 0.5, w: 0.1, h: 0.1 };
    let b = Box4 { cx: 0.9, cy: 0.5, w: 0.1, h: 0.1 };
    let g = giou(a, b);
    assert!(g < 0.0, "no-overlap boxes → GIoU < 0, got {g}");
    assert!(g >= -1.0, "GIoU ≥ −1, got {g}");
}

#[test]
fn test_giou_partial_overlap() {
    // 50% overlap in x, full overlap in y
    let a = Box4 { cx: 0.25, cy: 0.5, w: 0.5, h: 0.4 };
    let b = Box4 { cx: 0.5,  cy: 0.5, w: 0.5, h: 0.4 };
    let g = giou(a, b);
    assert!(g > 0.0 && g < 1.0, "partial overlap → GIoU ∈ (0, 1), got {g}");
}

// ── Hungarian matcher ─────────────────────────────────────────────────────────

#[test]
fn test_matcher_single_gt() {
    // 3 queries, 1 GT of class 0
    // Query 1 has very high logit for class 0 and a near-perfect box
    let n_queries = 3;
    let n_classes = 2;

    // Logits: [3, 2] — query 1 strongly predicts class 0
    let logits = vec![
        -10.0f32,  0.0,   // query 0: low prob for class 0
         10.0f32,  0.0,   // query 1: high prob for class 0
         -5.0f32,  3.0,   // query 2: predicts class 1
    ];

    // Boxes: [3, 4] cx/cy/w/h
    let boxes = vec![
        0.9f32, 0.9, 0.1, 0.1, // query 0: far off
        0.5f32, 0.5, 0.4, 0.4, // query 1: close to GT
        0.1f32, 0.1, 0.1, 0.1, // query 2: far off
    ];

    // GT: class 0, box near (0.5, 0.5, 0.4, 0.4)
    let gt_classes = vec![0usize];
    let gt_boxes   = vec![0.5f32, 0.5, 0.4, 0.4];

    let matches = hungarian_match(
        &logits, &boxes, &gt_classes, &gt_boxes,
        n_queries, n_classes, MatchWeights::default(),
    );

    assert_eq!(matches.len(), 1, "one GT → one match");
    let (q_idx, gt_idx) = matches[0];
    assert_eq!(gt_idx, 0, "matched GT index should be 0");
    assert_eq!(q_idx, 1, "query 1 should be matched (best class + box)");
}

#[test]
fn test_matcher_two_gt() {
    // 4 queries, 2 GTs
    // Query 0 is best for GT 0 (class 0), query 3 is best for GT 1 (class 1)
    let n_queries = 4;
    let n_classes = 2;

    let logits = vec![
        10.0f32, -5.0,   // q0: class 0
        -3.0f32,  1.0,   // q1: weakly class 1
        -2.0f32,  2.0,   // q2: class 1
        -5.0f32, 10.0,   // q3: class 1
    ];
    let boxes = vec![
        0.2f32, 0.2, 0.3, 0.3, // q0 → GT 0
        0.8f32, 0.8, 0.1, 0.1, // q1: off
        0.7f32, 0.7, 0.1, 0.1, // q2: off
        0.7f32, 0.7, 0.2, 0.2, // q3 → GT 1
    ];
    let gt_classes = vec![0usize, 1usize];
    let gt_boxes   = vec![
        0.2f32, 0.2, 0.3, 0.3, // GT 0
        0.7f32, 0.7, 0.2, 0.2, // GT 1
    ];

    let mut matches = hungarian_match(
        &logits, &boxes, &gt_classes, &gt_boxes,
        n_queries, n_classes, MatchWeights::default(),
    );
    matches.sort_by_key(|&(_, gt)| gt);

    assert_eq!(matches.len(), 2, "two GTs → two matches");
    assert_eq!(matches[0], (0, 0), "q0 ↔ GT0");
    assert_eq!(matches[1], (3, 1), "q3 ↔ GT1");
}

#[test]
fn test_matcher_empty_gt() {
    let matches = hungarian_match(
        &[], &[], &[], &[], 4, 2, MatchWeights::default(),
    );
    assert!(matches.is_empty(), "empty GT → no matches");
}
