// BCE classification loss forward test.
//
// Verifies the yolo_bce_cls_loss_forward kernel against PyTorch-generated
// reference outputs (fixtures/yolo_bce_cls_loss/).
//
// Pred / target layout: [C, N] channels-first.
// Output: [N] per-anchor BCE loss summed over C classes.

use std::path::PathBuf;
use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, errors::Result, testing};

const N:       usize = 32;
const C:       usize = 80;
const BLOCK_N: i32   = 32;

fn load(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

#[test]
fn test_yolo_bce_cls_loss_forward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::yolo::kernels::loss::cls::YoloBceClsLossForward::new(BLOCK_N);
    let target = Target::new(Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("yolo_bce_cls_loss_forward_source", kernel.source());
    assert_debug_snapshot!("yolo_bce_cls_loss_forward_mlir", mlir.trim());
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_yolo_bce_cls_loss_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let pred     = load("yolo_bce_cls_loss/pred.bin");
    let target   = load("yolo_bce_cls_loss/target.bin");
    let expected = load("yolo_bce_cls_loss/expected.bin");

    assert_eq!(pred.len(),     C * N);
    assert_eq!(target.len(),   C * N);
    assert_eq!(expected.len(), N);

    let mut pred_buf   = device.buffer::<f32>(C * N)?;
    let mut target_buf = device.buffer::<f32>(C * N)?;
    let loss_buf       = device.buffer::<f32>(N)?;

    pred_buf.to_device(&pred)?;
    target_buf.to_device(&target)?;

    let kernel = vision_rs::models::yolo::kernels::loss::cls::YoloBceClsLossForward::new(BLOCK_N);
    let cuda_target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &cuda_target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::yolo::kernels::loss::cls::YoloBceClsLossForward
    >(&ptx)?;

    let n_tiles = N.div_ceil(BLOCK_N as usize);
    let cfg = testing::launch_config_with_grid(n_tiles, &program);
    device.launch(&program, &cfg, (
        pred_buf.as_device_ptr() as *mut f32,
        target_buf.as_device_ptr() as *mut f32,
        loss_buf.as_device_ptr() as *mut f32,
        N as i32,
        C as i32,
    ))?;

    let mut loss_host = vec![0.0f32; N];
    loss_buf.to_host(&mut loss_host)?;

    for i in 0..N {
        assert!(
            (loss_host[i] - expected[i]).abs() < 1e-3,
            "mismatch at anchor {i}: gpu={:.6}  expected={:.6}",
            loss_host[i], expected[i]
        );
    }
    Ok(())
}
