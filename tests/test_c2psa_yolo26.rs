/*
 * SpinorML Ltd 🚀 AGPL-3.0 License - https://spinorml.com/license
 */

// C2PSA graph-structure tests.
//
// C2PSA(c1, c2, n, e) architecture:
//   cv1:   Conv2d + BN + SiLU  (c_in → 2*c)
//   split: ChannelChunk × 2    (→ a [B,c,H,W] and b [B,c,H,W])
//   for each PSABlock (× n):
//     Attention (qkv + FA2 + pe + proj)
//     Add (attn residual)
//     ffn0: Conv2d + BN + SiLU (c → 2c)
//     ffn1: Conv2d + BN        (2c → c)
//     Add (ffn residual)
//   ChannelCat (a, b)
//   cv2:   Conv2d + BN + SiLU  (2c → c_out)
//
// Shape contract: output is [B, c_out, H, W] for any spatial size.
// Graph contract: op counts verify correct wiring (a pure identity stub
//   cannot pass c_in==c_out shape checks, but the Attention count catches it).

use teeny_core::graph::{DtypeRepr, Op, SymTensor};
use vision_rs::models::yolo::yolo26::blocks::c2psa::c2psa;

fn sym_input(c: usize, h: usize, w: usize) -> SymTensor {
    let (t, _graph) =
        SymTensor::input(DtypeRepr::F32, vec![Some(2), Some(c), Some(h), Some(w)]);
    t
}

fn count_op(y: &SymTensor, pred: impl Fn(&Op) -> bool) -> usize {
    y.graph.borrow().nodes.iter().filter(|n| pred(&n.op)).count()
}

// ── n=1 (standard YOLO26 C2PSA) ──────────────────────────────────────────────
//
// C2PSA(256, 256, n=1, e=0.5): c=128, num_heads=2, key_dim=32.
// Add nodes:      2 (attn residual + ffn residual)
// ChannelChunk:   2 (split cv1 output into a and b)
// ChannelCat:     1 (cat a and b before cv2)
// Attention:      1

#[test]
fn test_c2psa_n1_output_shape() {
    let x = sym_input(256, 16, 16);
    let y = c2psa::<f32>(256, 256, 1, 0.5)(x);
    assert_eq!(y.shape, vec![Some(2), Some(256), Some(16), Some(16)]);
}

#[test]
fn test_c2psa_n1_graph_structure() {
    let x = sym_input(256, 16, 16);
    let y = c2psa::<f32>(256, 256, 1, 0.5)(x);

    assert_eq!(count_op(&y, |op| matches!(op, Op::Attention { .. })), 1);
    assert_eq!(count_op(&y, |op| matches!(op, Op::Add)),              2);
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelChunk { .. })), 2);
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelCat { .. })),   1);
}

// ── n=2 ───────────────────────────────────────────────────────────────────────
//
// Two PSABlocks: Add × 4, Attention × 2, ChannelChunk × 2, ChannelCat × 1.

#[test]
fn test_c2psa_n2_graph_structure() {
    let x = sym_input(256, 16, 16);
    let y = c2psa::<f32>(256, 256, 2, 0.5)(x);

    assert_eq!(y.shape, vec![Some(2), Some(256), Some(16), Some(16)]);
    assert_eq!(count_op(&y, |op| matches!(op, Op::Attention { .. })), 2);
    assert_eq!(count_op(&y, |op| matches!(op, Op::Add)),              4);
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelChunk { .. })), 2);
    assert_eq!(count_op(&y, |op| matches!(op, Op::ChannelCat { .. })),   1);
}

// ── Attention op carries correct parameters ───────────────────────────────────

#[test]
fn test_c2psa_attention_params() {
    let x = sym_input(256, 16, 16);
    let y = c2psa::<f32>(256, 256, 1, 0.5)(x);

    let g = y.graph.borrow();
    let attn_node = g.nodes.iter().find(|n| matches!(n.op, Op::Attention { .. })).unwrap();
    assert!(matches!(
        attn_node.op,
        Op::Attention { c: 128, num_heads: 2, key_dim: 32 }
    ));
    // Attention preserves spatial shape [B, c, H, W].
    assert_eq!(attn_node.shape, vec![Some(2), Some(128), Some(16), Some(16)]);
}
