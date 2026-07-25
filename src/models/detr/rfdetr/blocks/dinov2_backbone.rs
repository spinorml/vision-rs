/*
 * Copyright 2026 Teenygrad
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


//! DINOv2 ViT backbone for RF-DETR.
//!
//! Pipeline (all shapes for a single call, B = batch):
//!
//!   Input:         [B, 3, H_img, W_img]
//!   patch_embed:   Conv2d(3→D, k=patch_size, stride=patch_size) → [B, D, H_p, W_p]
//!   nchw_to_nld:   custom kernel → [B, N, D]  (N = H_p * W_p)
//!   add_pos_embed: custom kernel → [B, N, D]  (broadcast + pos[N, D] param)
//!   cat_cls:       custom kernel → [B, N+1, D] (prepend cls_token param)
//!   × depth        vit_block     → [B, N+1, D]
//!   remove_cls:    custom kernel → [B, N, D]
//!   nld_to_nchw:   custom kernel → [B, D, H_p, W_p]
//!
//! The output is suitable for a multi-scale feature pyramid or direct regression head.

use teeny_core::{
    dtype::Float,
    graph::{CustomData, SymTensor},
    name_scope::name_scope,
    nn::{Layer, conv2d::Conv2d},
};

use crate::models::detr::rfdetr::kernels::{
    cls_embed::{Dinov2AddPosEmbedOp, Dinov2CatClsOp, Dinov2RemoveClsOp},
    reshape::{Dinov2NchwToNldOp, Dinov2NldToNchwOp},
};

use super::vit_block::vit_block;

// ── DINOv2 Backbone ───────────────────────────────────────────────────────────

/// DINOv2 ViT backbone.
///
/// Parameters (auto-allocated by teenygrad runtime):
/// - `patch_embed.weight`: `[D, 3, patch_size, patch_size]`
/// - `patch_embed.bias`:   `[D]`  (if `patch_bias`)
/// - `pos_embed`:          `[N, D]`  (N = (img_h/patch_size) * (img_w/patch_size))
/// - `cls_token`:          `[D]`
/// - Per-block params (ln1, qkv, proj, ln2, fc1, fc2) × `depth`
///
/// Input:  `[B, 3, img_h, img_w]`
/// Output: `[B, D, img_h/patch_size, img_w/patch_size]`
pub fn dinov2_backbone<D: Float + 'static>(
    embed_dim:  usize,
    depth:      usize,
    num_heads:  usize,
    mlp_ratio:  usize,
    patch_size: usize,
    img_h:      usize,
    img_w:      usize,
) -> impl Fn(SymTensor) -> SymTensor {
    let h_patches = img_h / patch_size;
    let w_patches = img_w / patch_size;
    let d = embed_dim as i32;

    let patch_embed = Conv2d::<D, SymTensor, SymTensor, 4>::new(
        3, embed_dim,
        (patch_size, patch_size),
        (patch_size, patch_size),
        (0, 0),
        true,
    );

    let blocks: Vec<_> = (0..depth)
        .map(|_| vit_block::<D>(embed_dim, num_heads, mlp_ratio))
        .collect();

    move |x: SymTensor| {
        // ── Patch embed ───────────────────────────────────────────────────────
        // [B, 3, H_img, W_img] → [B, D, H_p, W_p]
        let x = { let _g = name_scope("patch_embed"); patch_embed.call(x) };

        // ── NCHW → NLD ───────────────────────────────────────────────────────
        // [B, D, H_p, W_p] → [B, N, D]
        let x = x.record_custom(
            CustomData::new(Dinov2NchwToNldOp::new(d)),
            &[],
            None,
        );

        // ── Positional embedding ──────────────────────────────────────────────
        // [B, N, D] → [B, N, D]  (param: pos_embed [N, D])
        let x = x.record_custom(
            CustomData::new(Dinov2AddPosEmbedOp::new(d)),
            &[],
            None,
        );

        // ── Prepend class token ───────────────────────────────────────────────
        // [B, N, D] → [B, N+1, D]  (param: cls_token [D])
        let x = x.record_custom(
            CustomData::new(Dinov2CatClsOp::new(d)),
            &[],
            None,
        );

        // ── Transformer blocks ────────────────────────────────────────────────
        let x = blocks.iter().enumerate().fold(x, |acc, (i, block)| {
            let _g = name_scope(&format!("block_{i}"));
            block(acc)
        });

        // ── Remove class token ────────────────────────────────────────────────
        // [B, N+1, D] → [B, N, D]
        let x = x.record_custom(
            CustomData::new(Dinov2RemoveClsOp::new(d)),
            &[],
            None,
        );

        // ── NLD → NCHW ───────────────────────────────────────────────────────
        // [B, N, D] → [B, D, H_p, W_p]
        x.record_custom(
            CustomData::new(Dinov2NldToNchwOp::new(d, h_patches, w_patches)),
            &[],
            None,
        )
    }
}
