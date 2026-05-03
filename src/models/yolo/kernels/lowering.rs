//! YoloLowering — middleware lowering that handles YOLO custom ops and
//! delegates all other ops to [`TritonLowering`].

use std::sync::Arc;

use anyhow::anyhow;
use teeny_core::{
    errors::Result,
    graph::{Graph, Op},
    model::{ExecutableOp, Lowering, LoweringMode},
    utils::dag::Dag,
};
use teeny_kernels::graph::{KernelExecutable, TritonLowering};

use super::detect_decode::{DetectDecodeForward, DetectDecodeOp, DetectDecodeRuntimeOp};

/// Lowering for vision-rs models.
///
/// Handles `Op::Custom` nodes registered by vision-rs (currently
/// `"yolo.detect_decode"`).  All other ops — including any `Op::Custom` nodes
/// with unknown names — are delegated to the base [`TritonLowering`].
pub struct YoloLowering {
    base: TritonLowering,
}

impl Default for YoloLowering {
    fn default() -> Self {
        Self::new()
    }
}

impl YoloLowering {
    pub fn new() -> Self {
        Self { base: TritonLowering::new() }
    }
}

impl<'a> Lowering<'a> for YoloLowering {
    fn lower(&self, graph: &Graph, mode: LoweringMode) -> Result<Dag<Box<dyn ExecutableOp>>> {
        // Identify all Op::Custom nodes this lowering handles.
        let custom_nodes: Vec<(usize, Arc<dyn teeny_core::graph::CustomOp>)> = graph.nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| {
                if let Op::Custom { data } = &node.op {
                    if data.name() == "yolo.detect_decode" {
                        return Some((idx, data.0.clone()));
                    }
                }
                None
            })
            .collect();

        if custom_nodes.is_empty() {
            // Nothing for us to handle — delegate entirely to the base.
            return self.base.lower(graph, mode);
        }

        // Build a modified graph where detect_decode nodes are replaced by
        // Op::Relu (shape-preserving) so TritonLowering can lower the rest.
        let mut modified = graph.clone();
        for &(ci, _) in &custom_nodes {
            modified.nodes[ci].op = Op::Relu;
        }

        // Lower the modified graph and capture the graph→DAG index mapping.
        let (mut dag, graph_to_dag) = self.base.lower_with_mapping(&modified, mode)?;

        // Replace each Relu placeholder with the real detect_decode KernelExecutable.
        for (ci, custom_op) in &custom_nodes {
            let dd_op = custom_op
                .as_any()
                .downcast_ref::<DetectDecodeOp>()
                .ok_or_else(|| anyhow!("expected DetectDecodeOp at custom node {ci}"))?;

            const BLOCK_A: i32 = 16;
            let kernel = DetectDecodeForward::new(BLOCK_A);
            let runtime_op = Arc::new(DetectDecodeRuntimeOp::new(
                dd_op.anchor_x.clone(),
                dd_op.anchor_y.clone(),
                dd_op.strides.clone(),
                BLOCK_A,
            ));

            let dag_idx = graph_to_dag[*ci];
            dag.node_mut(dag_idx).value = Box::new(KernelExecutable {
                name: "detect_decode_forward".to_string(),
                kernel_source: kernel.source,
                entry_point: "entry_point".to_string(),
                shape: graph.nodes[*ci].shape.clone(),
                dtype: graph.nodes[*ci].dtype,
                #[cfg(feature = "training")]
                backward_kernel_source: String::new(),
                #[cfg(feature = "training")]
                backward_entry_point: "entry_point".to_string(),
                runtime_op,
            });
        }

        Ok(dag)
    }

    fn base_lowering(&self) -> Option<&dyn Lowering<'a>> {
        Some(&self.base)
    }
}
