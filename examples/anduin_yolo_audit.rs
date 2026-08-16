//! YOLO26n Anduin fusion audit.
//!
//!   cargo run --release --example anduin_yolo_audit --features cuda

use std::collections::BTreeMap;

use teeny_core::graph::{DtypeRepr, Graph, Op, SymTensor};
use teeny_kernels::graph::optimizer::{Anduin, GraphOptimizer};
use vision_rs::models::yolo::yolo26::{Yolo26Variant, blocks::detect::DetectHead, yolo26};

fn op_label(op: &Op) -> String {
    match op {
        Op::Custom { data } => format!("custom:{}", data.name()),
        other => {
            let s = format!("{other:?}");
            s.split('{').next().unwrap_or(&s).trim().to_string()
        }
    }
}

fn histogram(graph: &Graph) -> BTreeMap<String, usize> {
    let mut h = BTreeMap::new();
    for node in &graph.nodes {
        *h.entry(op_label(&node.op)).or_default() += 1;
    }
    h
}

fn main() -> anyhow::Result<()> {
    let (input_sym, _) = SymTensor::input(
        DtypeRepr::F32,
        vec![None, Some(3), Some(640), Some(640)],
    );
    let out = yolo26::<f32>(80, &Yolo26Variant::N, DetectHead::OneToOne)(input_sym);
    let before = out.boxes.graph.borrow().clone();
    let after = Anduin.optimize(&before)?;

    println!("YOLO26n BEFORE: {} nodes", before.nodes.len());
    for (k, v) in histogram(&before) {
        println!("  {v:>4}  {k}");
    }
    println!("\nYOLO26n AFTER:  {} nodes", after.nodes.len());
    for (k, v) in histogram(&after) {
        println!("  {v:>4}  {k}");
    }
    println!(
        "\nΔ nodes: {} → {} ({:+})",
        before.nodes.len(),
        after.nodes.len(),
        after.nodes.len() as i64 - before.nodes.len() as i64
    );
    Ok(())
}
