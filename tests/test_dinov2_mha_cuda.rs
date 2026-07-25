// DINOv2 MHA pack/unpack kernel tests.
//
// Verifies dinov2_pack_qkv (fwd+bwd) and dinov2_unpack_attn (fwd+bwd)
// against Python-generated fixtures.
//
// Tensor layouts (all row-major):
//   qkv:    [B, N, 3*H*HD]    — linear projection output
//   packed: [3*BH, N, HD]     — FA2 input layout (s=Q/K/V, bh=b*H+h)
//   attn:   [BH, N, HD]       — FA2 output
//   out:    [B, N, H*HD]      — unpacked output

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

const B:        usize = 2;
const H:        usize = 4;
const HEAD_DIM: i32   = 32;
const N:        usize = 16;
const BH:       usize = B * H;

fn load(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

// ── pack_qkv snapshots ────────────────────────────────────────────────────────

#[test]
fn test_dinov2_pack_qkv_forward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::detr::rfdetr::kernels::mha::Dinov2PackQkv::new(HEAD_DIM);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("dinov2_pack_qkv_forward_source", kernel.source());
    assert_debug_snapshot!("dinov2_pack_qkv_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_dinov2_pack_qkv_backward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::detr::rfdetr::kernels::mha::Dinov2PackQkvBackward::new(HEAD_DIM);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("dinov2_pack_qkv_backward_source", kernel.source());
    assert_debug_snapshot!("dinov2_pack_qkv_backward_mlir",   mlir.trim());
    Ok(())
}

// ── unpack_attn snapshots ─────────────────────────────────────────────────────

#[test]
fn test_dinov2_unpack_attn_forward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::detr::rfdetr::kernels::mha::Dinov2UnpackAttn::new(HEAD_DIM);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("dinov2_unpack_attn_forward_source", kernel.source());
    assert_debug_snapshot!("dinov2_unpack_attn_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_dinov2_unpack_attn_backward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::detr::rfdetr::kernels::mha::Dinov2UnpackAttnBackward::new(HEAD_DIM);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("dinov2_unpack_attn_backward_source", kernel.source());
    assert_debug_snapshot!("dinov2_unpack_attn_backward_mlir",   mlir.trim());
    Ok(())
}

// ── pack_qkv CUDA forward ─────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_dinov2_pack_qkv_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let qkv      = load("dinov2_pack_qkv/qkv.bin");
    let expected = load("dinov2_pack_qkv/expected_packed.bin");

    assert_eq!(qkv.len(),      B * N * 3 * H * HEAD_DIM as usize);
    assert_eq!(expected.len(), 3 * BH * N * HEAD_DIM as usize);

    let mut qkv_buf = device.buffer::<f32>(qkv.len())?;
    let out_buf     = device.buffer::<f32>(expected.len())?;
    qkv_buf.to_device(&qkv)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::mha::Dinov2PackQkv::new(HEAD_DIM);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::mha::Dinov2PackQkv
    >(&ptx)?;

    // Grid: [3*BH*N, 1, 1]; one CTA per (section, batch-head, token)
    let cfg = CudaLaunchConfig {
        grid:    [(3 * BH * N) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        qkv_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N         as i32,   // n_ctx
        H         as i32,   // num_heads
        BH        as i32,   // bh_total
    ))?;

    let mut out_host = vec![0.0f32; expected.len()];
    out_buf.to_host(&mut out_host)?;

    let mut max_err = 0.0f32;
    for i in 0..expected.len() {
        let err = (out_host[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-5,
            "pack_qkv forward mismatch at {i}: gpu={:.7}  expected={:.7}  diff={:.2e}",
            out_host[i], expected[i], err
        );
    }
    println!("  pack_qkv forward max_err = {max_err:.2e}");
    Ok(())
}

// ── pack_qkv CUDA backward ────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_dinov2_pack_qkv_backward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let grad_packed = load("dinov2_pack_qkv/grad_packed.bin");
    let expected    = load("dinov2_pack_qkv/expected_dqkv.bin");

    assert_eq!(grad_packed.len(), 3 * BH * N * HEAD_DIM as usize);
    assert_eq!(expected.len(),    B * N * 3 * H * HEAD_DIM as usize);

    let mut gp_buf  = device.buffer::<f32>(grad_packed.len())?;
    let dqkv_buf    = device.buffer::<f32>(expected.len())?;
    gp_buf.to_device(&grad_packed)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::mha::Dinov2PackQkvBackward::new(HEAD_DIM);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::mha::Dinov2PackQkvBackward
    >(&ptx)?;

    let cfg = CudaLaunchConfig {
        grid:    [(3 * BH * N) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        gp_buf.as_device_ptr()   as *mut f32,  // d_packed_ptr
        dqkv_buf.as_device_ptr() as *mut f32,  // d_qkv_ptr
        N         as i32,   // n_ctx
        H         as i32,   // num_heads
        BH        as i32,   // bh_total
    ))?;

    let mut out_host = vec![0.0f32; expected.len()];
    dqkv_buf.to_host(&mut out_host)?;

    let mut max_err = 0.0f32;
    for i in 0..expected.len() {
        let err = (out_host[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-5,
            "pack_qkv backward mismatch at {i}: gpu={:.7}  expected={:.7}  diff={:.2e}",
            out_host[i], expected[i], err
        );
    }
    println!("  pack_qkv backward max_err = {max_err:.2e}");
    Ok(())
}

// ── unpack_attn CUDA forward ──────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_dinov2_unpack_attn_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let attn_out = load("dinov2_unpack_attn/attn_out.bin");
    let expected = load("dinov2_unpack_attn/expected_unpacked.bin");

    assert_eq!(attn_out.len(), BH * N * HEAD_DIM as usize);
    assert_eq!(expected.len(), B * N * H * HEAD_DIM as usize);

    let mut attn_buf = device.buffer::<f32>(attn_out.len())?;
    let out_buf      = device.buffer::<f32>(expected.len())?;
    attn_buf.to_device(&attn_out)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::mha::Dinov2UnpackAttn::new(HEAD_DIM);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::mha::Dinov2UnpackAttn
    >(&ptx)?;

    // Grid: [BH*N, 1, 1]
    let cfg = CudaLaunchConfig {
        grid:    [(BH * N) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        attn_buf.as_device_ptr() as *mut f32,  // attn_ptr
        out_buf.as_device_ptr()  as *mut f32,  // out_ptr
        N         as i32,   // n_ctx
        H         as i32,   // num_heads
        BH        as i32,   // _bh_total (unused but must be passed)
    ))?;

    let mut out_host = vec![0.0f32; expected.len()];
    out_buf.to_host(&mut out_host)?;

    let mut max_err = 0.0f32;
    for i in 0..expected.len() {
        let err = (out_host[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-5,
            "unpack_attn forward mismatch at {i}: gpu={:.7}  expected={:.7}  diff={:.2e}",
            out_host[i], expected[i], err
        );
    }
    println!("  unpack_attn forward max_err = {max_err:.2e}");
    Ok(())
}

// ── unpack_attn CUDA backward ─────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_dinov2_unpack_attn_backward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let grad_unpacked = load("dinov2_unpack_attn/grad_unpacked.bin");
    let expected      = load("dinov2_unpack_attn/expected_dattn_out.bin");

    assert_eq!(grad_unpacked.len(), B * N * H * HEAD_DIM as usize);
    assert_eq!(expected.len(),      BH * N * HEAD_DIM as usize);

    let mut gu_buf  = device.buffer::<f32>(grad_unpacked.len())?;
    let dattn_buf   = device.buffer::<f32>(expected.len())?;
    gu_buf.to_device(&grad_unpacked)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::mha::Dinov2UnpackAttnBackward::new(HEAD_DIM);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::mha::Dinov2UnpackAttnBackward
    >(&ptx)?;

    // Grid: [BH*N, 1, 1]
    let cfg = CudaLaunchConfig {
        grid:    [(BH * N) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        gu_buf.as_device_ptr()    as *mut f32,  // d_out_ptr
        dattn_buf.as_device_ptr() as *mut f32,  // d_attn_ptr
        N         as i32,   // n_ctx
        H         as i32,   // num_heads
        BH        as i32,   // _bh_total
    ))?;

    let mut out_host = vec![0.0f32; expected.len()];
    dattn_buf.to_host(&mut out_host)?;

    let mut max_err = 0.0f32;
    for i in 0..expected.len() {
        let err = (out_host[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-5,
            "unpack_attn backward mismatch at {i}: gpu={:.7}  expected={:.7}  diff={:.2e}",
            out_host[i], expected[i], err
        );
    }
    println!("  unpack_attn backward max_err = {max_err:.2e}");
    Ok(())
}
