// DINOv2 NCHW ↔ NLD reshape kernel tests.
//
// Verifies dinov2_nchw_to_nld (fwd+bwd) and dinov2_nld_to_nchw (fwd+bwd)
// against Python-generated fixtures.
//
// Tensor layouts:
//   nchw: [B, D, H, W]  — patch embedding output
//   nld:  [B, N, D]     — ViT sequence format  (N = H * W)

use std::path::PathBuf;
use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{device::CudaLaunchConfig, errors::Result, testing};

// ── Dimensions ────────────────────────────────────────────────────────────────

const B:   usize = 2;
const D:   i32   = 64;   // embed_dim
const H:   usize = 4;    // patch rows
const W:   usize = 4;    // patch cols
const N:   usize = H * W;

fn load(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

// ── nchw_to_nld snapshots ─────────────────────────────────────────────────────

#[test]
fn test_dinov2_nchw_to_nld_forward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::detr::rfdetr::kernels::reshape::Dinov2NchwToNld::new(D);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("dinov2_nchw_to_nld_forward_source", kernel.source());
    assert_debug_snapshot!("dinov2_nchw_to_nld_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_dinov2_nld_to_nchw_forward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::detr::rfdetr::kernels::reshape::Dinov2NldToNchw::new(D);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("dinov2_nld_to_nchw_forward_source", kernel.source());
    assert_debug_snapshot!("dinov2_nld_to_nchw_forward_mlir",   mlir.trim());
    Ok(())
}

// ── nchw_to_nld CUDA forward ──────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_dinov2_nchw_to_nld_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let nchw     = load("dinov2_reshape/nchw.bin");
    let expected = load("dinov2_reshape/expected_nld.bin");

    assert_eq!(nchw.len(),     B * D as usize * H * W);
    assert_eq!(expected.len(), B * N * D as usize);

    let mut in_buf  = device.buffer::<f32>(nchw.len())?;
    let out_buf     = device.buffer::<f32>(expected.len())?;
    in_buf.to_device(&nchw)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::reshape::Dinov2NchwToNld::new(D);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::reshape::Dinov2NchwToNld
    >(&ptx)?;

    let cfg = CudaLaunchConfig {
        grid:    [(B * N) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        in_buf.as_device_ptr()  as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,  // n_spatial
        D,         // embed_dim
    ))?;

    let mut out_host = vec![0.0f32; expected.len()];
    out_buf.to_host(&mut out_host)?;

    let mut max_err = 0.0f32;
    for i in 0..expected.len() {
        let err = (out_host[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-5,
            "nchw_to_nld mismatch at {i}: gpu={:.7}  expected={:.7}  diff={:.2e}",
            out_host[i], expected[i], err
        );
    }
    println!("  nchw_to_nld forward max_err = {max_err:.2e}");
    Ok(())
}

// ── nchw_to_nld CUDA backward (= nld_to_nchw forward) ────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_dinov2_nchw_to_nld_backward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let grad_nld = load("dinov2_reshape/grad_nld.bin");
    let expected = load("dinov2_reshape/expected_dnchw.bin");

    assert_eq!(grad_nld.len(), B * N * D as usize);
    assert_eq!(expected.len(), B * D as usize * H * W);

    let mut in_buf  = device.buffer::<f32>(grad_nld.len())?;
    let out_buf     = device.buffer::<f32>(expected.len())?;
    in_buf.to_device(&grad_nld)?;

    // Backward of nchw_to_nld = nld_to_nchw
    let kernel = vision_rs::models::detr::rfdetr::kernels::reshape::Dinov2NldToNchw::new(D);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::reshape::Dinov2NldToNchw
    >(&ptx)?;

    let cfg = CudaLaunchConfig {
        grid:    [(B * N) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        in_buf.as_device_ptr()  as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,  // n_spatial
        D,         // embed_dim
    ))?;

    let mut out_host = vec![0.0f32; expected.len()];
    out_buf.to_host(&mut out_host)?;

    let mut max_err = 0.0f32;
    for i in 0..expected.len() {
        let err = (out_host[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-5,
            "nchw_to_nld backward mismatch at {i}: gpu={:.7}  expected={:.7}  diff={:.2e}",
            out_host[i], expected[i], err
        );
    }
    println!("  nchw_to_nld backward max_err = {max_err:.2e}");
    Ok(())
}

// ── nld_to_nchw CUDA forward ──────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_dinov2_nld_to_nchw_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let nld      = load("dinov2_reshape/nld.bin");
    let expected = load("dinov2_reshape/expected_nchw.bin");

    assert_eq!(nld.len(),      B * N * D as usize);
    assert_eq!(expected.len(), B * D as usize * H * W);

    let mut in_buf  = device.buffer::<f32>(nld.len())?;
    let out_buf     = device.buffer::<f32>(expected.len())?;
    in_buf.to_device(&nld)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::reshape::Dinov2NldToNchw::new(D);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::reshape::Dinov2NldToNchw
    >(&ptx)?;

    let cfg = CudaLaunchConfig {
        grid:    [(B * N) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        in_buf.as_device_ptr()  as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,  // n_spatial
        D,         // embed_dim
    ))?;

    let mut out_host = vec![0.0f32; expected.len()];
    out_buf.to_host(&mut out_host)?;

    let mut max_err = 0.0f32;
    for i in 0..expected.len() {
        let err = (out_host[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-5,
            "nld_to_nchw mismatch at {i}: gpu={:.7}  expected={:.7}  diff={:.2e}",
            out_host[i], expected[i], err
        );
    }
    println!("  nld_to_nchw forward max_err = {max_err:.2e}");
    Ok(())
}

// ── nld_to_nchw CUDA backward (= nchw_to_nld forward) ────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_dinov2_nld_to_nchw_backward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let grad_nchw = load("dinov2_reshape/grad_nchw.bin");
    let expected  = load("dinov2_reshape/expected_dnld.bin");

    assert_eq!(grad_nchw.len(), B * D as usize * H * W);
    assert_eq!(expected.len(),  B * N * D as usize);

    let mut in_buf  = device.buffer::<f32>(grad_nchw.len())?;
    let out_buf     = device.buffer::<f32>(expected.len())?;
    in_buf.to_device(&grad_nchw)?;

    // Backward of nld_to_nchw = nchw_to_nld
    let kernel = vision_rs::models::detr::rfdetr::kernels::reshape::Dinov2NchwToNld::new(D);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::reshape::Dinov2NchwToNld
    >(&ptx)?;

    let cfg = CudaLaunchConfig {
        grid:    [(B * N) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        in_buf.as_device_ptr()  as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,  // n_spatial
        D,         // embed_dim
    ))?;

    let mut out_host = vec![0.0f32; expected.len()];
    out_buf.to_host(&mut out_host)?;

    let mut max_err = 0.0f32;
    for i in 0..expected.len() {
        let err = (out_host[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-5,
            "nld_to_nchw backward mismatch at {i}: gpu={:.7}  expected={:.7}  diff={:.2e}",
            out_host[i], expected[i], err
        );
    }
    println!("  nld_to_nchw backward max_err = {max_err:.2e}");
    Ok(())
}
