/*
 * SpinorML Ltd 🚀 AGPL-3.0 License - https://spinorml.com/license
 */

use teeny_core::{dtype::Float, graph::{Op, SymTensor}, name_scope::name_scope};

use super::conv::conv;

// ── Graph helper ──────────────────────────────────────────────────────────────

pub(super) fn elem_add(a: SymTensor, b: SymTensor) -> SymTensor {
    let shape = a.shape.clone();
    let node_id = a.graph.borrow_mut().add_node(
        Op::Add, vec![a.node_id, b.node_id], a.dtype, shape.clone(),
    );
    SymTensor { node_id, graph: a.graph.clone(), dtype: a.dtype, shape }
}

// ── Bottleneck variants ───────────────────────────────────────────────────────

/// `conv(c → c//2, k=3) → conv(c//2 → c, k=3)`.
///
/// Matches `ultralytics.nn.modules.block.Bottleneck(k=(3,3), e=0.5)` defaults.
pub fn bottleneck_std<D: Float + 'static>(c: usize, shortcut: bool) -> impl Fn(SymTensor) -> SymTensor {
    let c_inner = (c as f32 * 0.5) as usize;
    let cv1 = conv::<D>(c, c_inner, 3, 1);
    let cv2 = conv::<D>(c_inner, c, 3, 1);
    move |x: SymTensor| {
        let y = {
            let tmp = { let _g = name_scope("cv1"); cv1(x.clone()) };
            let _g = name_scope("cv2"); cv2(tmp)
        };
        if shortcut { elem_add(x, y) } else { y }
    }
}

/// `conv(c → c, k=3) → conv(c → c, k=3)`.
///
/// Used as the inner bottleneck inside `C3k` blocks (e=1.0).
pub fn bottleneck_3x3<D: Float + 'static>(c: usize, shortcut: bool) -> impl Fn(SymTensor) -> SymTensor {
    let cv1 = conv::<D>(c, c, 3, 1);
    let cv2 = conv::<D>(c, c, 3, 1);
    move |x: SymTensor| {
        let y = {
            let tmp = { let _g = name_scope("cv1"); cv1(x.clone()) };
            let _g = name_scope("cv2"); cv2(tmp)
        };
        if shortcut { elem_add(x, y) } else { y }
    }
}
