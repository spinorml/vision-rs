// Sigmoid focal loss forward and backward CUDA tests.
//
// Verifies sigmoid_focal_loss_forward and sigmoid_focal_loss_backward kernels
// against Python-generated reference outputs (fixtures/sigmoid_focal_loss/).

use std::path::PathBuf;
use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{errors::Result, testing};

const N:       usize = 128;
const BLOCK_N: i32   = 128;
const ALPHA:   f32   = 0.25;
const GAMMA:   f32   = 2.0;
const NUM_BOXES: f32 = 8.0;

fn load(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

// ── Forward snapshot ──────────────────────────────────────────────────────────

#[test]
fn test_sigmoid_focal_loss_forward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::detr::rfdetr::kernels::focal_loss::SigmoidFocalLossForward::new(BLOCK_N);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("sigmoid_focal_loss_forward_source", kernel.source());
    assert_debug_snapshot!("sigmoid_focal_loss_forward_mlir",   mlir.trim());
    Ok(())
}

// ── Backward snapshot ─────────────────────────────────────────────────────────

#[test]
fn test_sigmoid_focal_loss_backward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::detr::rfdetr::kernels::focal_loss::SigmoidFocalLossBackward::new(BLOCK_N);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("sigmoid_focal_loss_backward_source", kernel.source());
    assert_debug_snapshot!("sigmoid_focal_loss_backward_mlir",   mlir.trim());
    Ok(())
}

// ── Forward CUDA ──────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_sigmoid_focal_loss_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let logits   = load("sigmoid_focal_loss/logits.bin");
    let targets  = load("sigmoid_focal_loss/targets.bin");
    let expected = load("sigmoid_focal_loss/expected_loss.bin");

    assert_eq!(logits.len(),   N);
    assert_eq!(targets.len(),  N);
    assert_eq!(expected.len(), N);

    let mut logits_buf  = device.buffer::<f32>(N)?;
    let mut target_buf  = device.buffer::<f32>(N)?;
    let loss_buf        = device.buffer::<f32>(N)?;

    logits_buf.to_device(&logits)?;
    target_buf.to_device(&targets)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::focal_loss::SigmoidFocalLossForward::new(BLOCK_N);
    let cuda_target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &cuda_target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::focal_loss::SigmoidFocalLossForward
    >(&ptx)?;

    let n_tiles = N.div_ceil(BLOCK_N as usize);
    let cfg = testing::launch_config_with_grid(n_tiles, &program);

    device.launch(&program, &cfg, (
        logits_buf.as_device_ptr()  as *mut f32,
        target_buf.as_device_ptr()  as *mut f32,
        loss_buf.as_device_ptr()    as *mut f32,
        N        as i32,
        ALPHA,
        GAMMA,
        NUM_BOXES,
    ))?;

    let mut loss_host = vec![0.0f32; N];
    loss_buf.to_host(&mut loss_host)?;

    let mut max_err = 0.0f32;
    for i in 0..N {
        let err = (loss_host[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-5,
            "forward mismatch at {i}: gpu={:.7}  expected={:.7}  diff={:.2e}",
            loss_host[i], expected[i], err
        );
    }
    println!("  focal_loss forward max_err = {max_err:.2e}");
    Ok(())
}

// ── Backward CUDA ─────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_sigmoid_focal_loss_backward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let logits    = load("sigmoid_focal_loss/logits.bin");
    let targets   = load("sigmoid_focal_loss/targets.bin");
    let grad_loss = load("sigmoid_focal_loss/grad_loss.bin");
    let expected  = load("sigmoid_focal_loss/expected_dlogits.bin");

    assert_eq!(logits.len(),   N);
    assert_eq!(targets.len(),  N);
    assert_eq!(grad_loss.len(), N);
    assert_eq!(expected.len(), N);

    let mut logits_buf   = device.buffer::<f32>(N)?;
    let mut target_buf   = device.buffer::<f32>(N)?;
    let mut grad_buf     = device.buffer::<f32>(N)?;
    let d_logits_buf     = device.buffer::<f32>(N)?;

    logits_buf.to_device(&logits)?;
    target_buf.to_device(&targets)?;
    grad_buf.to_device(&grad_loss)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::focal_loss::SigmoidFocalLossBackward::new(BLOCK_N);
    let cuda_target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &cuda_target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::focal_loss::SigmoidFocalLossBackward
    >(&ptx)?;

    let n_tiles = N.div_ceil(BLOCK_N as usize);
    let cfg = testing::launch_config_with_grid(n_tiles, &program);

    device.launch(&program, &cfg, (
        logits_buf.as_device_ptr()   as *mut f32,
        target_buf.as_device_ptr()   as *mut f32,
        grad_buf.as_device_ptr()     as *mut f32,
        d_logits_buf.as_device_ptr() as *mut f32,
        N        as i32,
        ALPHA,
        GAMMA,
        NUM_BOXES,
    ))?;

    let mut d_logits_host = vec![0.0f32; N];
    d_logits_buf.to_host(&mut d_logits_host)?;

    let mut max_err = 0.0f32;
    for i in 0..N {
        let err = (d_logits_host[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-5,
            "backward mismatch at {i}: gpu={:.7}  expected={:.7}  diff={:.2e}",
            d_logits_host[i], expected[i], err
        );
    }
    println!("  focal_loss backward max_err = {max_err:.2e}");
    Ok(())
}
