/*
 * SpinorML Ltd 🚀 AGPL-3.0 License - https://spinorml.com/license
 */

use teeny_core::graph::{Op, SymTensor};

/// Channel-wise concatenation of N NCHW tensors.
///
/// All inputs must share the same B, H, W. Channels are concatenated in order.
/// Output shape: `[B, C0+C1+...+CN, H, W]`.
pub fn concat() -> impl Fn(Vec<SymTensor>) -> SymTensor {
    |tensors| {
        assert!(!tensors.is_empty(), "concat requires at least one input");
        let c_total: usize = tensors.iter()
            .map(|t| t.shape[1].expect("channel dim must be known"))
            .sum();
        let first = &tensors[0];
        let shape = vec![first.shape[0], Some(c_total), first.shape[2], first.shape[3]];
        let inputs: Vec<usize> = tensors.iter().map(|t| t.node_id).collect();
        let node_id = first.graph.borrow_mut().add_node(
            Op::ChannelCat { c_total }, inputs, first.dtype, shape.clone(),
        );
        SymTensor { node_id, graph: first.graph.clone(), dtype: first.dtype, shape }
    }
}
