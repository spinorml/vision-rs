/*
 * SpinorML Ltd 🚀 AGPL-3.0 License - https://spinorml.com/license
 */

// Full backward-pass integration test for YOLO26n.
//
// Pipeline under test:
//   1. Trace YOLO26n graph (64×64 input, nc=80).
//   2. Compile all kernels (inference-mode BN has a complete backward).
//   3. Load model; initialise all parameters to 0.01.
//   4. forward_train → boxes [1,4·84] + scores [1,80·84] on device.
//   5. Yolo26Loss::compute_grads → (d_boxes, d_scores) on host.
//   6. Upload grads as TensorRef seeds; call backward_multi.
//   7. Assert at least one parameter gradient is non-zero.
//
// Notes:
//   - Inference-mode BatchNorm is used (frozen running stats = zeros / ones
//     at init).  Gradients still flow end-to-end through all other ops.
//   - Kernel compilation is cached; first run is slow, subsequent runs fast.
//   - Requires TEENY_RUSTC_PATH (and optionally TEENY_CACHE_DIR) in env.

#[cfg(feature = "cuda")]
mod cuda {
    use dotenv::dotenv;
    use serial_test::serial;
    use teeny_compiler::compiler::{backend::llvm::compiler::LlvmCompiler, target::cuda::Target};
    use teeny_core::{graph::{DtypeRepr, SymTensor}, model::LoweringMode};
    use teeny_cuda::{compiler::graph::CudaGraphCompiler, model::TensorRef, testing};
    use teeny_kernels::graph::TritonLowering;
    use vision_rs::models::yolo::{
        loss::yolo26::Yolo26Loss,
        yolo26::{Yolo26Variant, yolo26},
    };

    const BATCH:     usize = 1;
    const NC:        usize = 80;
    const IMG_H:     usize = 64;
    const IMG_W:     usize = 64;
    // Anchors at 64×64 with strides [8,16,32]:
    //   8×8 + 4×4 + 2×2 = 64 + 16 + 4 = 84
    const N_ANCHORS: usize = 84;

    #[test]
    #[serial]
    fn test_yolo26n_backward_grads_non_zero() -> anyhow::Result<()> {
        dotenv().ok();

        // ── CUDA setup ────────────────────────────────────────────────────────

        let env = testing::setup_cuda_env()?;
        let device = &env.device;
        let target = Target::new(env.capability);

        // ── 1. Trace YOLO26n graph ────────────────────────────────────────────

        let (input_sym, _graph_rc) = SymTensor::input(
            DtypeRepr::F32,
            vec![None, Some(3), Some(IMG_H), Some(IMG_W)],
        );
        let out = yolo26::<f32>(NC, &Yolo26Variant::N)(input_sym);
        // Both out.boxes and out.scores share the same Rc<RefCell<Graph>>.
        let graph = out.boxes.graph.borrow();

        // ── 2. Compile ────────────────────────────────────────────────────────

        let rustc_path = std::env::var("TEENY_RUSTC_PATH")
            .expect("TEENY_RUSTC_PATH must be set to run this test");
        let cache_dir = std::env::var("TEENY_CACHE_DIR")
            .unwrap_or_else(|_| "/tmp/teenygrad_rustc".to_string());
        let compiler = LlvmCompiler::new(rustc_path, cache_dir)?;
        let graph_compiler = CudaGraphCompiler::new(compiler);
        let lowering = TritonLowering::new();
        println!("compiling YOLO26n (first run may take several minutes)...");
        let cuda_model = graph_compiler.compile_model(
            &graph, &lowering, &target, LoweringMode::Inference, false,
        )?;
        println!("compilation complete ({} DAG nodes)", cuda_model.dag.len());

        // ── 3. Load + initialise parameters ──────────────────────────────────

        let mut model = cuda_model.load(device, BATCH)?;
        let param_info: Vec<(usize, Vec<Vec<usize>>)> = model
            .param_info()
            .map(|(idx, shapes)| (idx, shapes.to_vec()))
            .collect();
        for (idx, shapes) in &param_info {
            for (pi, shape) in shapes.iter().enumerate() {
                let n: usize = shape.iter().product();
                model.load_param_f32(*idx, pi, &vec![0.01f32; n])?;
            }
        }

        // ── 4. Forward pass ───────────────────────────────────────────────────

        let input_data = vec![1.0f32; BATCH * 3 * IMG_H * IMG_W];
        let input_ref = TensorRef::from_host_f32(&input_data, vec![BATCH, 3, IMG_H, IMG_W])?;

        model.zero_grad();
        // input_ref ptr is stored in the ActivationCache (freed on cache drop).
        let (_, cache) = model.forward_train(device, BATCH, &[input_ref])?;

        // ── 5. Identify terminal nodes + read predictions to host ─────────────

        let terminals = model.terminal_node_indices_sorted_by_size();
        assert_eq!(terminals.len(), 2,
            "expected exactly 2 terminal nodes (boxes + scores), got {}",
            terminals.len());
        let (boxes_idx, scores_idx) = (terminals[0], terminals[1]);

        let boxes_host  = cache.tensors[boxes_idx ].as_ref().unwrap().to_host_f32()?;
        let scores_host = cache.tensors[scores_idx].as_ref().unwrap().to_host_f32()?;

        assert_eq!(boxes_host.len(),  BATCH *  4 * N_ANCHORS,
            "boxes tensor has wrong size");
        assert_eq!(scores_host.len(), BATCH * NC * N_ANCHORS,
            "scores tensor has wrong size");

        // ── 6. Compute CIoU + BCE loss gradients ─────────────────────────────

        let loss    = Yolo26Loss::new(IMG_H, IMG_W, NC, env.capability);
        let gt_boxes = vec![vec![[32.0f32, 32.0, 20.0, 20.0]]];
        let gt_cls   = vec![vec![0usize]];
        let (d_boxes, d_scores) = loss.compute_grads(
            device, &boxes_host, &scores_host, &gt_boxes, &gt_cls,
        )?;

        // ── 7. Seed backward_multi and run the backward pass ─────────────────

        let d_boxes_ref  = TensorRef::from_host_f32(&d_boxes,  vec![BATCH,  4 * N_ANCHORS])?;
        let d_scores_ref = TensorRef::from_host_f32(&d_scores, vec![BATCH, NC * N_ANCHORS])?;

        model.backward_multi(
            device, BATCH,
            &[(boxes_idx, d_boxes_ref.clone()), (scores_idx, d_scores_ref.clone())],
            &cache,
        )?;

        d_boxes_ref .free()?;
        d_scores_ref.free()?;
        drop(cache);

        // ── 8. Verify non-zero gradients ──────────────────────────────────────

        let any_nonzero = param_info.iter().any(|(idx, shapes)| {
            shapes.iter().enumerate().any(|(pi, _)| {
                model.read_param_grad_f32(*idx, pi)
                    .map(|g| g.iter().any(|&v| v != 0.0))
                    .unwrap_or(false)
            })
        });
        assert!(
            any_nonzero,
            "all parameter gradients are zero — backward pass is not propagating"
        );

        // Count how many param nodes received a gradient.
        let grad_nodes: usize = param_info.iter().filter(|(idx, shapes)| {
            shapes.iter().enumerate().any(|(pi, _)| {
                model.read_param_grad_f32(*idx, pi)
                    .map(|g| g.iter().any(|&v| v != 0.0))
                    .unwrap_or(false)
            })
        }).count();
        println!("✓  YOLO26n backward: {grad_nodes}/{} param nodes have non-zero gradients",
            param_info.len());

        Ok(())
    }
}
