// DINOv2 class-token and positional-embedding kernel tests.
//
// Verifies dinov2_cat_cls, dinov2_add_pos_embed, dinov2_remove_cls
// (forward and backward) against Python-generated fixtures.

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
const N:   usize = 16;
const D:   i32   = 64;

fn load(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

// ── Snapshots ─────────────────────────────────────────────────────────────────

#[test]
fn test_dinov2_cat_cls_forward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::detr::rfdetr::kernels::cls_embed::Dinov2CatCls::new(D);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("dinov2_cat_cls_forward_source", kernel.source());
    assert_debug_snapshot!("dinov2_cat_cls_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_dinov2_add_pos_embed_forward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::detr::rfdetr::kernels::cls_embed::Dinov2AddPosEmbed::new(D);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("dinov2_add_pos_embed_forward_source", kernel.source());
    assert_debug_snapshot!("dinov2_add_pos_embed_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_dinov2_remove_cls_forward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::detr::rfdetr::kernels::cls_embed::Dinov2RemoveCls::new(D);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("dinov2_remove_cls_forward_source", kernel.source());
    assert_debug_snapshot!("dinov2_remove_cls_forward_mlir",   mlir.trim());
    Ok(())
}

// ── cat_cls CUDA forward ──────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_dinov2_cat_cls_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let tokens   = load("dinov2_cls_embed/cat_cls/tokens.bin");
    let cls      = load("dinov2_cls_embed/cat_cls/cls.bin");
    let expected = load("dinov2_cls_embed/cat_cls/expected_out.bin");

    assert_eq!(tokens.len(),   B * N * D as usize);
    assert_eq!(cls.len(),      D as usize);
    assert_eq!(expected.len(), B * (N + 1) * D as usize);

    let mut tok_buf = device.buffer::<f32>(tokens.len())?;
    let mut cls_buf = device.buffer::<f32>(cls.len())?;
    let out_buf     = device.buffer::<f32>(expected.len())?;
    tok_buf.to_device(&tokens)?;
    cls_buf.to_device(&cls)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::cls_embed::Dinov2CatCls::new(D);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::cls_embed::Dinov2CatCls
    >(&ptx)?;

    // Grid: [B * (N+1), 1, 1]
    let cfg = CudaLaunchConfig {
        grid:    [(B * (N + 1)) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        tok_buf.as_device_ptr() as *mut f32,
        cls_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,  // n_seq
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
            "cat_cls forward mismatch at {i}: gpu={:.7}  expected={:.7}  diff={:.2e}",
            out_host[i], expected[i], err
        );
    }
    println!("  cat_cls forward max_err = {max_err:.2e}");
    Ok(())
}

// ── cat_cls CUDA backward ─────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_dinov2_cat_cls_backward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let grad_out     = load("dinov2_cls_embed/cat_cls/grad_out.bin");
    let exp_dtokens  = load("dinov2_cls_embed/cat_cls/expected_dtokens.bin");
    let exp_dcls     = load("dinov2_cls_embed/cat_cls/expected_dcls.bin");

    let mut gout_buf = device.buffer::<f32>(grad_out.len())?;
    let dtok_buf     = device.buffer::<f32>(exp_dtokens.len())?;
    let dcls_buf     = device.buffer::<f32>(D as usize)?;  // zeroed for atomic_add
    gout_buf.to_device(&grad_out)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::cls_embed::Dinov2CatClsBackward::new(D);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::cls_embed::Dinov2CatClsBackward
    >(&ptx)?;

    let cfg = CudaLaunchConfig {
        grid:    [(B * (N + 1)) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        gout_buf.as_device_ptr()  as *mut f32,
        dtok_buf.as_device_ptr()  as *mut f32,
        dcls_buf.as_device_ptr()  as *mut f32,
        N as i32,
        D,  // embed_dim
    ))?;

    let mut dtok_host = vec![0.0f32; exp_dtokens.len()];
    let mut dcls_host = vec![0.0f32; D as usize];
    dtok_buf.to_host(&mut dtok_host)?;
    dcls_buf.to_host(&mut dcls_host)?;

    let mut max_err = 0.0f32;
    for i in 0..exp_dtokens.len() {
        let err = (dtok_host[i] - exp_dtokens[i]).abs();
        max_err = max_err.max(err);
        assert!(err < 1e-5,
            "cat_cls bwd dtokens mismatch at {i}: gpu={:.7}  exp={:.7}  diff={:.2e}",
            dtok_host[i], exp_dtokens[i], err);
    }
    for i in 0..D as usize {
        let err = (dcls_host[i] - exp_dcls[i]).abs();
        max_err = max_err.max(err);
        assert!(err < 5e-4,    // atomic_add, slight looser
            "cat_cls bwd dcls mismatch at {i}: gpu={:.7}  exp={:.7}  diff={:.2e}",
            dcls_host[i], exp_dcls[i], err);
    }
    println!("  cat_cls backward max_err = {max_err:.2e}");
    Ok(())
}

// ── add_pos_embed CUDA forward ────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_dinov2_add_pos_embed_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let tokens   = load("dinov2_cls_embed/add_pos/tokens.bin");
    let pos      = load("dinov2_cls_embed/add_pos/pos.bin");
    let expected = load("dinov2_cls_embed/add_pos/expected_out.bin");

    let mut tok_buf = device.buffer::<f32>(tokens.len())?;
    let mut pos_buf = device.buffer::<f32>(pos.len())?;
    let out_buf     = device.buffer::<f32>(expected.len())?;
    tok_buf.to_device(&tokens)?;
    pos_buf.to_device(&pos)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::cls_embed::Dinov2AddPosEmbed::new(D);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::cls_embed::Dinov2AddPosEmbed
    >(&ptx)?;

    let cfg = CudaLaunchConfig {
        grid:    [(B * N) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        tok_buf.as_device_ptr() as *mut f32,
        pos_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,
        D,  // embed_dim
    ))?;

    let mut out_host = vec![0.0f32; expected.len()];
    out_buf.to_host(&mut out_host)?;

    let mut max_err = 0.0f32;
    for i in 0..expected.len() {
        let err = (out_host[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(err < 1e-5,
            "add_pos_embed fwd mismatch at {i}: gpu={:.7}  exp={:.7}  diff={:.2e}",
            out_host[i], expected[i], err);
    }
    println!("  add_pos_embed forward max_err = {max_err:.2e}");
    Ok(())
}

// ── remove_cls CUDA forward ───────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_dinov2_remove_cls_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let tokens   = load("dinov2_cls_embed/remove_cls/tokens.bin");
    let expected = load("dinov2_cls_embed/remove_cls/expected_out.bin");

    let mut tok_buf = device.buffer::<f32>(tokens.len())?;
    let out_buf     = device.buffer::<f32>(expected.len())?;
    tok_buf.to_device(&tokens)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::cls_embed::Dinov2RemoveCls::new(D);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::cls_embed::Dinov2RemoveCls
    >(&ptx)?;

    let cfg = CudaLaunchConfig {
        grid:    [(B * N) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        tok_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,  // n_seq (patch count, NOT N+1)
        D,         // embed_dim
    ))?;

    let mut out_host = vec![0.0f32; expected.len()];
    out_buf.to_host(&mut out_host)?;

    let mut max_err = 0.0f32;
    for i in 0..expected.len() {
        let err = (out_host[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(err < 1e-5,
            "remove_cls fwd mismatch at {i}: gpu={:.7}  exp={:.7}  diff={:.2e}",
            out_host[i], expected[i], err);
    }
    println!("  remove_cls forward max_err = {max_err:.2e}");
    Ok(())
}
