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


//! DINOv2 ViT transformer block.
//!
//! One block:
//!   LayerNorm → QKV linear → pack_qkv → FlashAttn2 → unpack_attn → proj linear
//!   → Add residual → LayerNorm → MLP (linear → GELU → linear) → Add residual.
//!
//! Input/output shape: `[B, N, embed_dim]`  (sequence format, NLD).

use teeny_core::{
    dtype::Float,
    graph::{CustomData, SymTensor},
    name_scope::name_scope,
    nn::{Layer, activation::gelu::Gelu, layernorm::LayerNorm, linear::Linear},
};

use crate::models::detr::rfdetr::kernels::mha::{
    Dinov2PackQkvOp, Dinov2UnpackAttnOp, FlashAttn2Dinov2Op,
};

// ── Helper: element-wise add ──────────────────────────────────────────────────

fn add(a: SymTensor, b: SymTensor) -> SymTensor {
    use teeny_core::graph::Op;
    let shape = a.shape.clone();
    let node_id = a.graph.borrow_mut().add_node(
        Op::Add, vec![a.node_id, b.node_id], a.dtype, shape.clone(),
    );
    SymTensor { node_id, graph: a.graph.clone(), dtype: a.dtype, shape }
}

// ── DINOv2 ViT Block ─────────────────────────────────────────────────────────

/// Single DINOv2 ViT block (attention + MLP).
///
/// Parameters (via teenygrad's runtime):
/// - ln1: weight `[D]`, bias `[D]`
/// - qkv: weight `[3D, D]`, bias `[3D]`
/// - proj: weight `[D, D]`, bias `[D]`
/// - ln2: weight `[D]`, bias `[D]`
/// - mlp_fc1: weight `[mlp_dim, D]`, bias `[mlp_dim]`
/// - mlp_fc2: weight `[D, mlp_dim]`, bias `[D]`
/// - flash_attn_l: logsumexp scratch `[BH * N]` (auto-allocated by FlashAttn2Dinov2Op)
///
/// Input/output: `[B, N, D]`.
pub fn vit_block<D: Float + 'static>(
    embed_dim: usize,
    num_heads: usize,
    mlp_ratio: usize,
) -> impl Fn(SymTensor) -> SymTensor {
    let head_dim = embed_dim / num_heads;
    let mlp_dim  = embed_dim * mlp_ratio;

    let ln1  = LayerNorm::<D, SymTensor, SymTensor, 3>::new([embed_dim]);
    let qkv  = Linear::<D, SymTensor, SymTensor, 3>::new(embed_dim, 3 * embed_dim, true);
    let proj = Linear::<D, SymTensor, SymTensor, 3>::new(embed_dim, embed_dim, true);
    let ln2  = LayerNorm::<D, SymTensor, SymTensor, 3>::new([embed_dim]);
    let fc1  = Linear::<D, SymTensor, SymTensor, 3>::new(embed_dim, mlp_dim, true);
    let fc2  = Linear::<D, SymTensor, SymTensor, 3>::new(mlp_dim, embed_dim, true);
    let gelu = Gelu::<D, SymTensor, 3>::new();

    move |x: SymTensor| {
        let residual = x.clone();

        // ── Attention sub-block ────────────────────────────────────────────────
        let x = { let _g = name_scope("ln1"); ln1.call(x) };

        // [B, N, D] → [B, N, 3*D] (combined QKV projection)
        let qkv_out = { let _g = name_scope("qkv"); qkv.call(x) };

        // [B, N, 3*H*HD] → [3*BH, N, HD]
        let packed = qkv_out.record_custom(
            CustomData::new(Dinov2PackQkvOp::new(head_dim as i32, num_heads)),
            &[],
            None,
        );

        // [3*BH, N, HD] → [BH, N, HD]
        let attn_out = packed.record_custom(
            CustomData::new(FlashAttn2Dinov2Op::new(head_dim as i32)),
            &[],
            None,
        );

        // [BH, N, HD] → [B, N, H*HD] = [B, N, D]
        let unpacked = attn_out.record_custom(
            CustomData::new(Dinov2UnpackAttnOp::new(head_dim as i32, num_heads)),
            &[],
            None,
        );

        // Output projection
        let attn_proj = { let _g = name_scope("proj"); proj.call(unpacked) };

        // Residual add
        let x = add(residual, attn_proj);

        // ── MLP sub-block ──────────────────────────────────────────────────────
        let residual = x.clone();

        let x = { let _g = name_scope("ln2"); ln2.call(x) };
        let x = { let _g = name_scope("fc1"); fc1.call(x) };
        let x = { let _g = name_scope("gelu"); gelu.call(x) };
        let x = { let _g = name_scope("fc2"); fc2.call(x) };

        add(residual, x)
    }
}
