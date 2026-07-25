// Multi-Scale Deformable Attention forward and backward CUDA tests.
//
// Verifies ms_deform_attn_forward and ms_deform_attn_backward kernels against
// Python-generated fixtures (tests/fixtures/ms_deform_attn/).
//
// Tensor layouts (all row-major):
//   value:          [BH, S_total, HEAD_DIM]
//   sampling_locs:  [BH, NQ, n_levels, n_points, 2]
//   attn_weights:   [BH, NQ, n_levels * n_points]
//   spatial_shapes: [n_levels, 2]  (H_l, W_l as f32)
//   level_start:    [n_levels]     (cumulative starts as f32)
//   output:         [BH, NQ, HEAD_DIM]

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

const BH:        usize = 4;
const NQ:        usize = 6;
const N_LEVELS:  usize = 2;
const N_POINTS:  usize = 4;
const HEAD_DIM:  i32   = 8;
const S_TOTAL:   usize = 20;  // 4*4 + 2*2

fn load(rel: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("missing {path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

// ── Forward snapshot ──────────────────────────────────────────────────────────

#[test]
fn test_ms_deform_attn_forward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::detr::rfdetr::kernels::ms_deform_attn::MsDeformAttnForward::new(HEAD_DIM);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("ms_deform_attn_forward_source", kernel.source());
    assert_debug_snapshot!("ms_deform_attn_forward_mlir",   mlir.trim());
    Ok(())
}

// ── Backward snapshot ─────────────────────────────────────────────────────────

#[test]
fn test_ms_deform_attn_backward_snapshot() -> std::result::Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let kernel = vision_rs::models::detr::rfdetr::kernels::ms_deform_attn::MsDeformAttnBackward::new(HEAD_DIM);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm90);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;
    assert_debug_snapshot!("ms_deform_attn_backward_source", kernel.source());
    assert_debug_snapshot!("ms_deform_attn_backward_mlir",   mlir.trim());
    Ok(())
}

// ── Forward CUDA ──────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_ms_deform_attn_forward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let value          = load("ms_deform_attn/value.bin");
    let sampling_locs  = load("ms_deform_attn/sampling_locs.bin");
    let attn_weights   = load("ms_deform_attn/attn_weights.bin");
    let spatial_shapes = load("ms_deform_attn/spatial_shapes.bin");
    let level_start    = load("ms_deform_attn/level_start.bin");
    let expected       = load("ms_deform_attn/expected_output.bin");

    let nlp = N_LEVELS * N_POINTS;

    assert_eq!(value.len(),         BH * S_TOTAL * HEAD_DIM as usize);
    assert_eq!(sampling_locs.len(), BH * NQ * N_LEVELS * N_POINTS * 2);
    assert_eq!(attn_weights.len(),  BH * NQ * nlp);
    assert_eq!(spatial_shapes.len(), N_LEVELS * 2);
    assert_eq!(level_start.len(),    N_LEVELS);
    assert_eq!(expected.len(),       BH * NQ * HEAD_DIM as usize);

    let mut value_buf    = device.buffer::<f32>(value.len())?;
    let mut slocs_buf    = device.buffer::<f32>(sampling_locs.len())?;
    let mut aw_buf       = device.buffer::<f32>(attn_weights.len())?;
    let mut ss_buf       = device.buffer::<f32>(spatial_shapes.len())?;
    let mut ls_buf       = device.buffer::<f32>(level_start.len())?;
    let out_buf          = device.buffer::<f32>(expected.len())?;

    value_buf.to_device(&value)?;
    slocs_buf.to_device(&sampling_locs)?;
    aw_buf.to_device(&attn_weights)?;
    ss_buf.to_device(&spatial_shapes)?;
    ls_buf.to_device(&level_start)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::ms_deform_attn::MsDeformAttnForward::new(HEAD_DIM);
    let cuda_target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &cuda_target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::ms_deform_attn::MsDeformAttnForward
    >(&ptx)?;

    // Grid: (NQ, BH, 1); Block: threads from PTX metadata (.reqntid).
    // Triton compiles with num_warps=4 (128 threads) regardless of HEAD_DIM.
    let cfg = CudaLaunchConfig {
        grid:    [NQ as u32, BH as u32, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        value_buf.as_device_ptr()   as *mut f32,
        slocs_buf.as_device_ptr()   as *mut f32,
        aw_buf.as_device_ptr()      as *mut f32,
        ss_buf.as_device_ptr()      as *mut f32,
        ls_buf.as_device_ptr()      as *mut f32,
        out_buf.as_device_ptr()     as *mut f32,
        NQ        as i32,
        S_TOTAL   as i32,
        N_LEVELS  as i32,
        N_POINTS  as i32,
    ))?;

    let mut out_host = vec![0.0f32; expected.len()];
    out_buf.to_host(&mut out_host)?;

    let n = expected.len();
    let mut max_err = 0.0f32;
    for i in 0..n {
        let err = (out_host[i] - expected[i]).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-4,
            "forward mismatch at element {i}: gpu={:.7}  expected={:.7}  diff={:.2e}",
            out_host[i], expected[i], err
        );
    }
    println!("  forward max_err = {max_err:.2e}");
    Ok(())
}

// ── Backward CUDA ─────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "cuda")]
fn test_ms_deform_attn_backward_cuda() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let value          = load("ms_deform_attn/value.bin");
    let sampling_locs  = load("ms_deform_attn/sampling_locs.bin");
    let attn_weights   = load("ms_deform_attn/attn_weights.bin");
    let spatial_shapes = load("ms_deform_attn/spatial_shapes.bin");
    let level_start    = load("ms_deform_attn/level_start.bin");
    let grad_output    = load("ms_deform_attn/grad_output.bin");
    let exp_dvalue     = load("ms_deform_attn/expected_dvalue.bin");
    let exp_dslocs     = load("ms_deform_attn/expected_dsampling_locs.bin");
    let exp_daw        = load("ms_deform_attn/expected_dattn_weights.bin");

    let mut value_buf    = device.buffer::<f32>(value.len())?;
    let mut slocs_buf    = device.buffer::<f32>(sampling_locs.len())?;
    let mut aw_buf       = device.buffer::<f32>(attn_weights.len())?;
    let mut ss_buf       = device.buffer::<f32>(spatial_shapes.len())?;
    let mut ls_buf       = device.buffer::<f32>(level_start.len())?;
    let mut go_buf       = device.buffer::<f32>(grad_output.len())?;

    // Gradient output buffers (zeroed — d_value uses atomic_add)
    let dvalue_buf  = device.buffer::<f32>(value.len())?;
    let dslocs_buf  = device.buffer::<f32>(sampling_locs.len())?;
    let daw_buf     = device.buffer::<f32>(attn_weights.len())?;

    value_buf.to_device(&value)?;
    slocs_buf.to_device(&sampling_locs)?;
    aw_buf.to_device(&attn_weights)?;
    ss_buf.to_device(&spatial_shapes)?;
    ls_buf.to_device(&level_start)?;
    go_buf.to_device(&grad_output)?;

    let kernel = vision_rs::models::detr::rfdetr::kernels::ms_deform_attn::MsDeformAttnBackward::new(HEAD_DIM);
    let cuda_target = Target::new(env.capability);
    let ptx = std::fs::read(compile_kernel(&kernel, &cuda_target, true)?)?;
    let program = testing::load_program_from_ptx::<
        vision_rs::models::detr::rfdetr::kernels::ms_deform_attn::MsDeformAttnBackward
    >(&ptx)?;

    let cfg = CudaLaunchConfig {
        grid:    [NQ as u32, BH as u32, 1],
        block:   [program.threads_per_block(), 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    };

    device.launch(&program, &cfg, (
        value_buf.as_device_ptr()  as *mut f32,
        slocs_buf.as_device_ptr()  as *mut f32,
        aw_buf.as_device_ptr()     as *mut f32,
        ss_buf.as_device_ptr()     as *mut f32,
        ls_buf.as_device_ptr()     as *mut f32,
        go_buf.as_device_ptr()     as *mut f32,
        dvalue_buf.as_device_ptr()  as *mut f32,
        dslocs_buf.as_device_ptr()  as *mut f32,
        daw_buf.as_device_ptr()     as *mut f32,
        NQ       as i32,
        S_TOTAL  as i32,
        N_LEVELS as i32,
        N_POINTS as i32,
    ))?;

    let mut dvalue_host = vec![0.0f32; value.len()];
    let mut dslocs_host = vec![0.0f32; sampling_locs.len()];
    let mut daw_host    = vec![0.0f32; attn_weights.len()];
    dvalue_buf.to_host(&mut dvalue_host)?;
    dslocs_buf.to_host(&mut dslocs_host)?;
    daw_buf.to_host(&mut daw_host)?;

    // d_value check (accumulated via atomic_add, slightly looser tolerance)
    let mut max_err_dv = 0.0f32;
    for i in 0..value.len() {
        let err = (dvalue_host[i] - exp_dvalue[i]).abs();
        max_err_dv = max_err_dv.max(err);
        assert!(
            err < 5e-4,
            "d_value mismatch at {i}: gpu={:.7}  expected={:.7}  diff={:.2e}",
            dvalue_host[i], exp_dvalue[i], err
        );
    }

    // d_sampling_locs check
    let mut max_err_ds = 0.0f32;
    for i in 0..sampling_locs.len() {
        let err = (dslocs_host[i] - exp_dslocs[i]).abs();
        max_err_ds = max_err_ds.max(err);
        assert!(
            err < 1e-4,
            "d_sampling_locs mismatch at {i}: gpu={:.7}  expected={:.7}  diff={:.2e}",
            dslocs_host[i], exp_dslocs[i], err
        );
    }

    // d_attn_weights check
    let mut max_err_da = 0.0f32;
    for i in 0..attn_weights.len() {
        let err = (daw_host[i] - exp_daw[i]).abs();
        max_err_da = max_err_da.max(err);
        assert!(
            err < 1e-4,
            "d_attn_weights mismatch at {i}: gpu={:.7}  expected={:.7}  diff={:.2e}",
            daw_host[i], exp_daw[i], err
        );
    }

    println!("  backward max_err: d_value={max_err_dv:.2e}  d_slocs={max_err_ds:.2e}  d_aw={max_err_da:.2e}");
    Ok(())
}
