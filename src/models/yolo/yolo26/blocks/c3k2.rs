/*
 * SpinorML Ltd 🚀 AGPL-3.0 License - https://spinorml.com/license
 */

use teeny_core::{dtype::Float, graph::{Op, SymTensor}};

use super::conv::conv;

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

// ── Bottleneck variants ───────────────────────────────────────────────────────

/// conv(c, c//2, k=3) → conv(c//2, c, k=3), matches ultralytics Bottleneck defaults (k=(3,3), e=0.5).
fn bottleneck_std<D: Float>(c: usize, shortcut: bool) -> impl Fn(SymTensor) -> SymTensor {
    let c_inner = (c as f32 * 0.5) as usize;
    let cv1 = conv::<D>(c, c_inner, 3, 1);
    let cv2 = conv::<D>(c_inner, c, 3, 1);
    move |x: SymTensor| {
        let y = cv2(cv1(x.clone()));
        if shortcut { elem_add(x, y) } else { y }
    }
}

/// conv(c,c,k=3) → conv(c,c,k=3), used inside C3k blocks.
fn bottleneck_3x3<D: Float>(c: usize, shortcut: bool) -> impl Fn(SymTensor) -> SymTensor {
    let cv1 = conv::<D>(c, c, 3, 1);
    let cv2 = conv::<D>(c, c, 3, 1);
    move |x: SymTensor| {
        let y = cv2(cv1(x.clone()));
        if shortcut { elem_add(x, y) } else { y }
    }
}

// ── C3k inner block ───────────────────────────────────────────────────────────

/// Matches ultralytics C3k(c, c, n=2, shortcut, e=0.5).
fn c3k_inner<D: Float>(c: usize, shortcut: bool) -> impl Fn(SymTensor) -> SymTensor {
    let c_h = ((c as f32) * 0.5) as usize;
    let cv1 = conv::<D>(c, c_h, 1, 1);
    let cv2 = conv::<D>(c, c_h, 1, 1);
    let cv3 = conv::<D>(2 * c_h, c, 1, 1);
    let m0 = bottleneck_3x3::<D>(c_h, shortcut);
    let m1 = bottleneck_3x3::<D>(c_h, shortcut);
    move |x: SymTensor| {
        let h1 = m1(m0(cv1(x.clone())));
        let h2 = cv2(x);
        cv3(channel_cat(vec![h1, h2], 2 * c_h))
    }
}

// ── C3k2 block ────────────────────────────────────────────────────────────────

/// C3k2: the primary feature-extraction block in YOLO11/26.
///
/// Matches `ultralytics.nn.modules.block.C3k2` (which inherits from C2f).
/// Forward:
///   h = cv1(x)                          // [N, 2*c, H, W]
///   [y0, y1] = h.chunk(2, dim=1)        // each [N, c, H, W]
///   parts = [y0, y1]
///   last = y1
///   for each bottleneck b:
///       last = b(last); parts.append(last)
///   return cv2(cat(parts))              // [(2+n)*c → c_out]
pub fn c3k2<D: Float + 'static>(
    c_in: usize,
    c_out: usize,
    n: usize,
    c3k: bool,
    shortcut: bool,
    e: f32,
) -> impl Fn(SymTensor) -> SymTensor {
    let c = (c_out as f32 * e) as usize;
    let cv1 = conv::<D>(c_in, 2 * c, 1, 1);
    let cv2 = conv::<D>((2 + n) * c, c_out, 1, 1);
    let bottlenecks: Vec<Box<dyn Fn(SymTensor) -> SymTensor>> = (0..n)
        .map(|_| -> Box<dyn Fn(SymTensor) -> SymTensor> {
            if c3k {
                Box::new(c3k_inner::<D>(c, shortcut))
            } else {
                Box::new(bottleneck_std::<D>(c, shortcut))
            }
        })
        .collect();
    move |x: SymTensor| {
        let h = cv1(x);
        let y0 = channel_chunk(h.clone(), 2 * c, c, 0);
        let y1 = channel_chunk(h, 2 * c, c, c);
        let mut last = y1.clone();
        let mut parts = vec![y0, y1];
        for b in &bottlenecks {
            last = b(last);
            parts.push(last.clone());
        }
        cv2(channel_cat(parts, (2 + n) * c))
    }
}
