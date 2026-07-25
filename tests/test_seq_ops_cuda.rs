// Seq-ops kernel tests.
//
// Verifies pack_heads and seq_cat2 (forward and backward) against Python fixtures.

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

const B:       usize = 2;
const S:       usize = 16;
const N_HEADS: usize = 4;
const HD:      i32   = 32;
const D:       i32   = (N_HEADS as i32) * HD; // 128

const SA: usize = 10;
const SB: usize = 6;

fn load(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

// ── Snapshots ─────────────────────────────────────────────────────────────────

#[test]
fn test_pack_heads_forward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::detr::rfdetr::kernels::seq_ops::PackHeads::new(HD);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("pack_heads_forward_source", kernel.source());
    assert_debug_snapshot!("pack_heads_forward_mlir",   mlir.trim());
    Ok(())
}

#[test]
fn test_seq_cat2_forward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::detr::rfdetr::kernels::seq_ops::SeqCat2::new(D);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("seq_cat2_forward_source", kernel.source());
    assert_debug_snapshot!("seq_cat2_forward_mlir",   mlir.trim());
    Ok(())
}

// ── pack_heads CUDA forward ───────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_pack_heads_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let inp      = load("seq_ops/pack_heads/inp.bin");
    let expected = load("seq_ops/pack_heads/expected_out.bin");

    assert_eq!(inp.len(),      B * S * D as usize);
    assert_eq!(expected.len(), B * N_HEADS * S * HD as usize);

    let mut inp_buf = device.buffer::<f32>(inp.len())?;
    let out_buf     = device.buffer::<f32>(expected.len())?;
    inp_buf.to_device(&inp)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::seq_ops::PackHeads::new(HD);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::seq_ops::PackHeads
    >(&ptx)?;

    // Grid: [BH * S, 1, 1]
    let bh = B * N_HEADS;
    let cfg = CudaLaunchConfig {
        grid:    [(bh * S) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        inp_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N_HEADS as i32,
        S as i32,
        HD,
    ))?;

    let mut out_host = vec![0.0f32; expected.len()];
    out_buf.to_host(&mut out_host)?;

    let mut max_err = 0.0f32;
    for i in 0..expected.len() {
        let err = (out_host[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-5,
            "pack_heads fwd mismatch at {i}: gpu={:.7}  exp={:.7}  diff={:.2e}",
            out_host[i], expected[i], err
        );
    }
    println!("  pack_heads forward max_err = {max_err:.2e}");
    Ok(())
}

// ── pack_heads CUDA backward ──────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_pack_heads_backward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let grad_out  = load("seq_ops/pack_heads/grad_out.bin");
    let exp_dinp  = load("seq_ops/pack_heads/expected_dinp.bin");

    assert_eq!(grad_out.len(), B * N_HEADS * S * HD as usize);
    assert_eq!(exp_dinp.len(), B * S * D as usize);

    let mut gout_buf = device.buffer::<f32>(grad_out.len())?;
    let dinp_buf     = device.buffer::<f32>(exp_dinp.len())?;
    gout_buf.to_device(&grad_out)?;

    // Backward = unpack_heads
    let kernel = vision_rs::models::detr::rfdetr::kernels::seq_ops::UnpackHeads::new(HD);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::seq_ops::UnpackHeads
    >(&ptx)?;

    let bh = B * N_HEADS;
    let cfg = CudaLaunchConfig {
        grid:    [(bh * S) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        gout_buf.as_device_ptr()  as *mut f32,
        dinp_buf.as_device_ptr()  as *mut f32,
        N_HEADS as i32,
        S as i32,
        HD,
    ))?;

    let mut dinp_host = vec![0.0f32; exp_dinp.len()];
    dinp_buf.to_host(&mut dinp_host)?;

    let mut max_err = 0.0f32;
    for i in 0..exp_dinp.len() {
        let err = (dinp_host[i] - exp_dinp[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-5,
            "pack_heads bwd mismatch at {i}: gpu={:.7}  exp={:.7}  diff={:.2e}",
            dinp_host[i], exp_dinp[i], err
        );
    }
    println!("  pack_heads backward max_err = {max_err:.2e}");
    Ok(())
}

// ── seq_cat2 CUDA forward ─────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_seq_cat2_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let a        = load("seq_ops/seq_cat2/a.bin");
    let b        = load("seq_ops/seq_cat2/b.bin");
    let expected = load("seq_ops/seq_cat2/expected_out.bin");

    assert_eq!(a.len(),        B * SA * D as usize);
    assert_eq!(b.len(),        B * SB * D as usize);
    assert_eq!(expected.len(), B * (SA + SB) * D as usize);

    let mut a_buf = device.buffer::<f32>(a.len())?;
    let mut b_buf = device.buffer::<f32>(b.len())?;
    let out_buf   = device.buffer::<f32>(expected.len())?;
    a_buf.to_device(&a)?;
    b_buf.to_device(&b)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::seq_ops::SeqCat2::new(D);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::seq_ops::SeqCat2
    >(&ptx)?;

    let cfg = CudaLaunchConfig {
        grid:    [(B * (SA + SB)) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        a_buf.as_device_ptr() as *mut f32,
        b_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        SA as i32,
        SB as i32,
    ))?;

    let mut out_host = vec![0.0f32; expected.len()];
    out_buf.to_host(&mut out_host)?;

    let mut max_err = 0.0f32;
    for i in 0..expected.len() {
        let err = (out_host[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-5,
            "seq_cat2 fwd mismatch at {i}: gpu={:.7}  exp={:.7}  diff={:.2e}",
            out_host[i], expected[i], err
        );
    }
    println!("  seq_cat2 forward max_err = {max_err:.2e}");
    Ok(())
}

// ── seq_cat2 CUDA backward ────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_seq_cat2_backward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let grad_out = load("seq_ops/seq_cat2/grad_out.bin");
    let exp_da   = load("seq_ops/seq_cat2/expected_da.bin");
    let exp_db   = load("seq_ops/seq_cat2/expected_db.bin");

    let mut gout_buf = device.buffer::<f32>(grad_out.len())?;
    let da_buf       = device.buffer::<f32>(exp_da.len())?;
    let db_buf       = device.buffer::<f32>(exp_db.len())?;
    gout_buf.to_device(&grad_out)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::seq_ops::SeqSplit2::new(D);
    let target = Target::new(env.capability);
    let ptx    = std::fs::read(compile_kernel(&kernel, &target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::seq_ops::SeqSplit2
    >(&ptx)?;

    let cfg = CudaLaunchConfig {
        grid:    [(B * (SA + SB)) as u32, 1, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        gout_buf.as_device_ptr() as *mut f32,
        da_buf.as_device_ptr()   as *mut f32,
        db_buf.as_device_ptr()   as *mut f32,
        SA as i32,
        SB as i32,
    ))?;

    let mut da_host = vec![0.0f32; exp_da.len()];
    let mut db_host = vec![0.0f32; exp_db.len()];
    da_buf.to_host(&mut da_host)?;
    db_buf.to_host(&mut db_host)?;

    let mut max_err = 0.0f32;
    for i in 0..exp_da.len() {
        let err = (da_host[i] - exp_da[i]).abs();
        max_err = max_err.max(err);
        assert!(err < 1e-5,
            "seq_split2 bwd da mismatch at {i}: gpu={:.7}  exp={:.7}  diff={:.2e}",
            da_host[i], exp_da[i], err);
    }
    for i in 0..exp_db.len() {
        let err = (db_host[i] - exp_db[i]).abs();
        max_err = max_err.max(err);
        assert!(err < 1e-5,
            "seq_split2 bwd db mismatch at {i}: gpu={:.7}  exp={:.7}  diff={:.2e}",
            db_host[i], exp_db[i], err);
    }
    println!("  seq_cat2 backward max_err = {max_err:.2e}");
    Ok(())
}
