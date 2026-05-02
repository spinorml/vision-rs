/*
 * SpinorML Ltd 🚀 AGPL-3.0 License - https://spinorml.com/license
 */

use teeny_core::{dtype::Float, graph::{Op, SymTensor}};

use super::conv::{conv, conv_bn};

// ── Graph helpers ─────────────────────────────────────────────────────────────

fn channel_chunk(x: SymTensor, c_total: usize, chunk_c: usize, chunk_offset: usize) -> SymTensor {
    let op = Op::ChannelChunk { c_total, chunk_c, chunk_offset };
    let shape = vec![x.shape[0], Some(chunk_c), x.shape[2], x.shape[3]];
    let node_id = x.graph.borrow_mut().add_node(op, vec![x.node_id], x.dtype, shape.clone());
    SymTensor { node_id, graph: x.graph.clone(), dtype: x.dtype, shape }
}

fn channel_cat(tensors: Vec<SymTensor>, c_total: usize) -> SymTensor {
    let first = &tensors[0];
    let shape = vec![first.shape[0], Some(c_total), first.shape[2], first.shape[3]];
    let inputs: Vec<usize> = tensors.iter().map(|t| t.node_id).collect();
    let node_id = first.graph.borrow_mut().add_node(
        Op::ChannelCat { c_total }, inputs, first.dtype, shape.clone(),
    );
    SymTensor { node_id, graph: first.graph.clone(), dtype: first.dtype, shape }
}

fn elem_add(a: SymTensor, b: SymTensor) -> SymTensor {
    let shape = a.shape.clone();
    let node_id = a.graph.borrow_mut().add_node(
        Op::Add, vec![a.node_id, b.node_id], a.dtype, shape.clone(),
    );
    SymTensor { node_id, graph: a.graph.clone(), dtype: a.dtype, shape }
}

/// Multi-head self-attention with Flash Attention 2 + position encoding.
///
/// Represents `Attention.forward(x)` from ultralytics PSABlock:
///   qkv conv → FA2 → pe depthwise conv → proj conv → (attn + pe) output.
///
/// Input:  [B, c, H, W]
/// Output: [B, c, H, W]   (residual add is applied by the caller)
fn psa_attention(c: usize, num_heads: usize, key_dim: usize)
    -> impl Fn(SymTensor) -> SymTensor
{
    move |x: SymTensor| {
        let shape = x.shape.clone();
        let node_id = x.graph.borrow_mut().add_node(
            Op::Attention { c, num_heads, key_dim },
            vec![x.node_id],
            x.dtype,
            shape.clone(),
        );
        SymTensor { node_id, graph: x.graph.clone(), dtype: x.dtype, shape }
    }
}

// ── PSABlock ──────────────────────────────────────────────────────────────────

/// Single PSABlock iteration — attention residual + FFN residual.
///
/// Matches `ultralytics.nn.modules.block.PSABlock(c, attn_ratio=0.5, num_heads, shortcut=True)`.
fn psa_block<D: Float + 'static>(c: usize, num_heads: usize, key_dim: usize)
    -> impl Fn(SymTensor) -> SymTensor
{
    let attn = psa_attention(c, num_heads, key_dim);
    let ffn0 = conv::<D>(c, 2 * c, 1, 1);
    let ffn1 = conv_bn::<D>(2 * c, c, 1, 1, 1);
    move |b: SymTensor| {
        let b = elem_add(b.clone(), attn(b));
        let ffn_out = ffn1(ffn0(b.clone()));
        elem_add(b, ffn_out)
    }
}

// ── C2PSA ─────────────────────────────────────────────────────────────────────

/// C2PSA: cross-stage partial network with PSA attention blocks.
///
/// Matches `ultralytics.nn.modules.block.C2PSA(c1, c2, n, e)`.
///
/// Forward:
///   h        = cv1(x)           // [B, 2*c, H, W]
///   [a, b]   = split(h, c)      // each [B, c, H, W]
///   b        = PSABlock(b) × n
///   output   = cv2(cat(a, b))   // [B, c2, H, W]
pub fn c2psa<D: Float + 'static>(
    c_in: usize,
    c_out: usize,
    n: usize,
    e: f32,
) -> impl Fn(SymTensor) -> SymTensor {
    assert_eq!(c_in, c_out, "C2PSA requires c_in == c_out");
    let c = (c_out as f32 * e) as usize;
    let num_heads = c / 64;
    let key_dim = 32;

    let cv1 = conv::<D>(c_in, 2 * c, 1, 1);
    let cv2 = conv::<D>(2 * c, c_out, 1, 1);
    let blocks: Vec<Box<dyn Fn(SymTensor) -> SymTensor>> = (0..n)
        .map(|_| -> Box<dyn Fn(SymTensor) -> SymTensor> {
            Box::new(psa_block::<D>(c, num_heads, key_dim))
        })
        .collect();

    move |x: SymTensor| {
        let h  = cv1(x);
        let a  = channel_chunk(h.clone(), 2 * c, c, 0);
        let mut b = channel_chunk(h, 2 * c, c, c);
        for blk in &blocks {
            b = blk(b);
        }
        cv2(channel_cat(vec![a, b], 2 * c))
    }
}
