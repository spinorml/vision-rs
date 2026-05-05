/*
 * SpinorML Ltd 🚀 AGPL-3.0 License - https://spinorml.com/license
 */

//! YOLO26 training loss: CIoU box regression + BCE classification.
//!
//! # Pipeline (per training step)
//!
//! 1. **Decode** raw LTRB predictions to XYWH using the anchor grid.
//! 2. **Assign** GT boxes to anchors (CPU-side TaskAlignedAssigner).
//! 3. **Box loss**: CIoU forward (saved activations) then backward.
//! 4. **Cls loss**: BCE cls backward on all anchors.
//! 5. **Decode backward**: d_xywh → d_ltrb through the linear decode.
//! 6. Return `d_boxes` `[B, 4*A]` and `d_scores` `[B, nc*A]` (host f32).

pub use super::{anchor::AnchorGrid, assign::TaskAlignedAssigner};

/// CPU reference CIoU forward — returns saved activations (iou, v, alpha).
pub fn ciou_forward_cpu(pred: &[f32], target: &[f32], n: usize)
    -> (Vec<f32>, Vec<f32>, Vec<f32>)
{
    const EPS: f32 = 1e-7;
    let pi2 = std::f32::consts::PI * std::f32::consts::PI;
    let mut iou   = vec![0.0f32; n];
    let mut v_vec = vec![0.0f32; n];
    let mut alpha = vec![0.0f32; n];

    for i in 0..n {
        let px = pred[i];           let py = pred[n + i];
        let pw = pred[2 * n + i];   let ph = pred[3 * n + i];
        let tx = target[i];         let ty = target[n + i];
        let tw = target[2 * n + i]; let th = target[3 * n + i];

        let px1 = px - pw * 0.5; let px2 = px + pw * 0.5;
        let py1 = py - ph * 0.5; let py2 = py + ph * 0.5;
        let tx1 = tx - tw * 0.5; let tx2 = tx + tw * 0.5;
        let ty1 = ty - th * 0.5; let ty2 = ty + th * 0.5;

        let inter = ((px2.min(tx2) - px1.max(tx1)).max(0.0))
                  * ((py2.min(ty2) - py1.max(ty1)).max(0.0));
        let union = pw * ph + tw * th - inter;
        let iou_i = inter / (union + EPS);

        let v_i = (4.0 / pi2)
            * ((tw / (th + EPS)).atan() - (pw / (ph + EPS)).atan()).powi(2);
        let alpha_i = v_i / (1.0 - iou_i + v_i + EPS);

        iou[i] = iou_i; v_vec[i] = v_i; alpha[i] = alpha_i;
    }
    (iou, v_vec, alpha)
}

// ── CUDA-only training loss ───────────────────────────────────────────────────

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
    use teeny_core::device::{Device, buffer::Buffer};
    use teeny_cuda::{compiler::target::Capability, device::CudaDevice, errors::Result, testing};
    use crate::models::yolo::kernels::loss::{
        ciou::{YoloCiouLossBackward, YoloCiouLossForward},
        cls::YoloBceClsLossBackward,
    };

    /// CUDA training loss for YOLO26: computes d_boxes and d_scores.
    pub struct Yolo26Loss {
        pub grid:     AnchorGrid,
        pub assigner: TaskAlignedAssigner,
        pub nc:       usize,
        pub cap:      Capability,
        block_n:      i32,
    }

    impl Yolo26Loss {
        pub fn new(img_h: usize, img_w: usize, nc: usize, cap: Capability) -> Self {
            Self { grid: AnchorGrid::yolo26(img_h, img_w), assigner: TaskAlignedAssigner::default(), nc, cap, block_n: 64 }
        }

        /// Compute loss gradients for one batch.
        ///
        /// `boxes`   – raw LTRB predictions `[B, 4*A]` (host f32, channels-first)
        /// `scores`  – class logits `[B, nc*A]` (host f32, channels-first)
        ///
        /// Returns `(d_boxes [B,4*A], d_scores [B,nc*A])` as host f32.
        pub fn compute_grads(
            &self,
            device: &CudaDevice<'_>,
            boxes:      &[f32],
            scores:     &[f32],
            gt_boxes_b: &[Vec<[f32; 4]>],
            gt_cls_b:   &[Vec<usize>],
        ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
            let b  = gt_boxes_b.len();
            let a  = self.grid.n_anchors;
            let nc = self.nc;

            assert_eq!(boxes.len(),  b * 4 * a);
            assert_eq!(scores.len(), b * nc * a);

            let target = Target::new(self.cap);
            let bn = self.block_n;

            // Compile kernels (cached by the compiler on disk).
            let ciou_bwd_ptx = std::fs::read(compile_kernel(&YoloCiouLossBackward::new(bn), &target, true)?)?;
            let cls_bwd_ptx  = std::fs::read(compile_kernel(&YoloBceClsLossBackward::new(bn), &target, true)?)?;
            // Forward needed for saved activations (iou, v, alpha from GPU).
            let ciou_fwd_ptx = std::fs::read(compile_kernel(&YoloCiouLossForward::new(bn), &target, true)?)?;

            let prog_ciou_fwd = testing::load_program_from_ptx::<YoloCiouLossForward>(&ciou_fwd_ptx)?;
            let prog_ciou_bwd = testing::load_program_from_ptx::<YoloCiouLossBackward>(&ciou_bwd_ptx)?;
            let prog_cls_bwd  = testing::load_program_from_ptx::<YoloBceClsLossBackward>(&cls_bwd_ptx)?;

            let mut d_boxes_out  = vec![0.0f32; b * 4 * a];
            let mut d_scores_out = vec![0.0f32; b * nc * a];

            for bi in 0..b {
                let boxes_i  = &boxes[bi * 4 * a .. (bi + 1) * 4 * a];
                let scores_i = &scores[bi * nc * a .. (bi + 1) * nc * a];

                // 1. Decode LTRB → XYWH (CPU).
                let xywh = self.grid.decode_ltrb_to_xywh(boxes_i);

                // 2. Assign GT targets (CPU).
                let assign = self.assigner.assign(
                    &xywh, scores_i,
                    &self.grid.cx, &self.grid.cy,
                    &gt_boxes_b[bi], &gt_cls_b[bi],
                );
                let n_pos: usize = assign.is_positive.iter().filter(|&&p| p).count();
                let loss_scale = if n_pos > 0 { 1.0 / n_pos as f32 } else { 0.0 };

                // 3. CIoU backward for positive anchors.
                let mut d_xywh = vec![0.0f32; 4 * a];
                if n_pos > 0 {
                    let pos_idx: Vec<usize> = (0..a).filter(|&i| assign.is_positive[i]).collect();
                    let np = pos_idx.len();

                    let mut pred_pos   = vec![0.0f32; 4 * np];
                    let mut target_pos = vec![0.0f32; 4 * np];
                    for (j, &i) in pos_idx.iter().enumerate() {
                        for ch in 0..4 {
                            pred_pos[ch * np + j]   = xywh[ch * a + i];
                            target_pos[ch * np + j] = assign.target_boxes[ch * a + i];
                        }
                    }

                    // CIoU forward on GPU to get saved activations.
                    let (iou, v, alpha) = self.ciou_fwd_gpu(
                        device, &prog_ciou_fwd, &pred_pos, &target_pos, np
                    )?;

                    let dy_pos: Vec<f32> = pos_idx.iter()
                        .map(|&i| loss_scale * assign.iou_weight[i])
                        .collect();

                    let d_pred_pos = self.ciou_bwd_gpu(
                        device, &prog_ciou_bwd,
                        &dy_pos, &pred_pos, &target_pos, &iou, &v, &alpha, np,
                    )?;

                    for (j, &i) in pos_idx.iter().enumerate() {
                        for ch in 0..4 { d_xywh[ch * a + i] = d_pred_pos[ch * np + j]; }
                    }
                }

                // 4. Decode backward: d_xywh → d_ltrb.
                let d_ltrb = self.grid.decode_backward(&d_xywh);
                d_boxes_out[bi * 4 * a .. (bi + 1) * 4 * a].copy_from_slice(&d_ltrb);

                // 5. BCE cls backward — all anchors.
                let mut cls_target = vec![0.0f32; nc * a];
                for (i, &pos) in assign.is_positive.iter().enumerate() {
                    if pos {
                        let c = assign.target_cls[i];
                        if c < nc { cls_target[c * a + i] = 1.0; }
                    }
                }
                let dy_cls = vec![loss_scale; a];

                let d_scores_i = self.cls_bwd_gpu(
                    device, &prog_cls_bwd,
                    &dy_cls, scores_i, &cls_target, a, nc,
                )?;
                d_scores_out[bi * nc * a .. (bi + 1) * nc * a].copy_from_slice(&d_scores_i);
            }

            Ok((d_boxes_out, d_scores_out))
        }

        // ── GPU helpers ───────────────────────────────────────────────────────

        fn ciou_fwd_gpu(
            &self, device: &CudaDevice<'_>,
            prog: &teeny_cuda::device::program::CudaProgram<'_, YoloCiouLossForward>,
            pred: &[f32], target: &[f32], n: usize,
        ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
            let mut pred_buf   = device.buffer::<f32>(4 * n)?;
            let mut target_buf = device.buffer::<f32>(4 * n)?;
            let loss_buf   = device.buffer::<f32>(n)?;
            let iou_buf    = device.buffer::<f32>(n)?;
            let v_buf      = device.buffer::<f32>(n)?;
            let alpha_buf  = device.buffer::<f32>(n)?;
            pred_buf.to_device(pred)?;
            target_buf.to_device(target)?;
            let cfg = testing::launch_config_with_grid(n.div_ceil(self.block_n as usize), prog);
            device.launch(prog, &cfg, (
                pred_buf.as_device_ptr()   as *mut f32,
                target_buf.as_device_ptr() as *mut f32,
                loss_buf.as_device_ptr()   as *mut f32,
                iou_buf.as_device_ptr()    as *mut f32,
                v_buf.as_device_ptr()      as *mut f32,
                alpha_buf.as_device_ptr()  as *mut f32,
                n as i32,
            ))?;
            let (mut iou, mut v, mut alpha) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
            iou_buf.to_host(&mut iou)?;
            v_buf.to_host(&mut v)?;
            alpha_buf.to_host(&mut alpha)?;
            Ok((iou, v, alpha))
        }

        fn ciou_bwd_gpu(
            &self, device: &CudaDevice<'_>,
            prog: &teeny_cuda::device::program::CudaProgram<'_, YoloCiouLossBackward>,
            dy: &[f32], pred: &[f32], target: &[f32],
            iou: &[f32], v: &[f32], alpha: &[f32], n: usize,
        ) -> Result<Vec<f32>> {
            let mut dy_buf     = device.buffer::<f32>(n)?;
            let mut pred_buf   = device.buffer::<f32>(4 * n)?;
            let mut target_buf = device.buffer::<f32>(4 * n)?;
            let mut iou_buf    = device.buffer::<f32>(n)?;
            let mut v_buf      = device.buffer::<f32>(n)?;
            let mut alpha_buf  = device.buffer::<f32>(n)?;
            let d_pred_buf     = device.buffer::<f32>(4 * n)?;
            dy_buf.to_device(dy)?; pred_buf.to_device(pred)?; target_buf.to_device(target)?;
            iou_buf.to_device(iou)?; v_buf.to_device(v)?; alpha_buf.to_device(alpha)?;
            let cfg = testing::launch_config_with_grid(n.div_ceil(self.block_n as usize), prog);
            device.launch(prog, &cfg, (
                dy_buf.as_device_ptr()     as *mut f32,
                pred_buf.as_device_ptr()   as *mut f32,
                target_buf.as_device_ptr() as *mut f32,
                iou_buf.as_device_ptr()    as *mut f32,
                v_buf.as_device_ptr()      as *mut f32,
                alpha_buf.as_device_ptr()  as *mut f32,
                d_pred_buf.as_device_ptr() as *mut f32,
                n as i32,
            ))?;
            let mut out = vec![0.0f32; 4 * n];
            d_pred_buf.to_host(&mut out)?;
            Ok(out)
        }

        fn cls_bwd_gpu(
            &self, device: &CudaDevice<'_>,
            prog: &teeny_cuda::device::program::CudaProgram<'_, YoloBceClsLossBackward>,
            dy: &[f32], pred: &[f32], target: &[f32],
            n: usize, c: usize,
        ) -> Result<Vec<f32>> {
            let mut dy_buf     = device.buffer::<f32>(n)?;
            let mut pred_buf   = device.buffer::<f32>(c * n)?;
            let mut target_buf = device.buffer::<f32>(c * n)?;
            let d_pred_buf     = device.buffer::<f32>(c * n)?;
            dy_buf.to_device(dy)?; pred_buf.to_device(pred)?; target_buf.to_device(target)?;
            let cfg = testing::launch_config_with_grid(n.div_ceil(self.block_n as usize), prog);
            device.launch(prog, &cfg, (
                dy_buf.as_device_ptr()     as *mut f32,
                pred_buf.as_device_ptr()   as *mut f32,
                target_buf.as_device_ptr() as *mut f32,
                d_pred_buf.as_device_ptr() as *mut f32,
                n as i32,
                c as i32,
            ))?;
            let mut out = vec![0.0f32; c * n];
            d_pred_buf.to_host(&mut out)?;
            Ok(out)
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::Yolo26Loss;
