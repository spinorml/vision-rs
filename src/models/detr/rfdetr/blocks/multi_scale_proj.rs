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


//! Multi-scale feature projector for RF-DETR.
//!
//! Takes DINOv2 backbone output `[B, D, H_p, W_p]` and produces three feature
//! maps at strides ×1 (s4), ×2 upsample (s3), and ×2 downsample (s5):
//!
//!   s3: upsample_2× + conv_bn_silu  → [B, out_ch, 2*H_p, 2*W_p]
//!   s4: identity   + conv_bn_silu  → [B, out_ch,   H_p,   W_p]
//!   s5: maxpool_2× + conv_bn_silu  → [B, out_ch, H_p/2, W_p/2]
//!
//! Each scale then goes through a C2f block for cross-stage partial fusion.

use teeny_core::{
    dtype::Float,
    graph::{Op, SymTensor},
    name_scope::name_scope,
    nn::{
        Layer,
        activation::sigmoid::Silu,
        batchnorm::BatchNorm2d,
        conv2d::Conv2d,
        pool::MaxPool2d,
    },
};

// ── Graph helpers ─────────────────────────────────────────────────────────────

fn channel_cat(tensors: &[SymTensor], c_total: usize) -> SymTensor {
    let first = &tensors[0];
    let mut b_shape = first.shape.clone();
    b_shape[1] = Some(c_total);
    let input_ids: Vec<usize> = tensors.iter().map(|t| t.node_id).collect();
    let node_id = first.graph.borrow_mut().add_node(
        Op::ChannelCat { c_total },
        input_ids,
        first.dtype,
        b_shape,
    );
    SymTensor {
        node_id,
        graph: first.graph.clone(),
        dtype: first.dtype,
        shape: tensors[0].shape.iter().enumerate().map(|(i, d)| {
            if i == 1 { Some(c_total) } else { *d }
        }).collect(),
    }
}

fn channel_chunk(x: &SymTensor, c_total: usize, chunk_c: usize, chunk_offset: usize) -> SymTensor {
    let mut shape = x.shape.clone();
    shape[1] = Some(chunk_c);
    let node_id = x.graph.borrow_mut().add_node(
        Op::ChannelChunk { c_total, chunk_c, chunk_offset },
        vec![x.node_id],
        x.dtype,
        shape.clone(),
    );
    SymTensor { node_id, graph: x.graph.clone(), dtype: x.dtype, shape }
}

fn upsample_2x(x: SymTensor) -> SymTensor {
    let mut shape = x.shape.clone();
    shape[2] = shape[2].map(|h| h * 2);
    shape[3] = shape[3].map(|w| w * 2);
    let node_id = x.graph.borrow_mut().add_node(
        Op::UpsampleNearest2d { scale_h: 2, scale_w: 2 },
        vec![x.node_id],
        x.dtype,
        shape.clone(),
    );
    SymTensor { node_id, graph: x.graph.clone(), dtype: x.dtype, shape }
}

fn add(a: SymTensor, b: SymTensor) -> SymTensor {
    let shape = a.shape.clone();
    let node_id = a.graph.borrow_mut().add_node(
        Op::Add,
        vec![a.node_id, b.node_id],
        a.dtype,
        shape.clone(),
    );
    SymTensor { node_id, graph: a.graph.clone(), dtype: a.dtype, shape }
}

// ── Building blocks ───────────────────────────────────────────────────────────

struct ConvBnSilu<D: Float> {
    conv: Conv2d<D, SymTensor, SymTensor, 4>,
    bn:   BatchNorm2d<D, SymTensor, SymTensor, 4>,
    silu: Silu<D, SymTensor, 4>,
}

impl<D: Float + 'static> ConvBnSilu<D> {
    fn new(in_ch: usize, out_ch: usize, k: usize, stride: usize, pad: usize) -> Self {
        Self {
            conv: Conv2d::new(in_ch, out_ch, (k, k), (stride, stride), (pad, pad), false),
            bn:   BatchNorm2d::new(out_ch),
            silu: Silu::new(),
        }
    }

    fn call(&self, x: SymTensor) -> SymTensor {
        let x = self.conv.call(x);
        let x = self.bn.call(x);
        self.silu.call(x)
    }
}

/// Bottleneck block: two 3×3 ConvBnSilu layers, optional shortcut.
struct Bottleneck<D: Float> {
    cv1: ConvBnSilu<D>,
    cv2: ConvBnSilu<D>,
    shortcut: bool,
}

impl<D: Float + 'static> Bottleneck<D> {
    fn new(c: usize, shortcut: bool) -> Self {
        Self {
            cv1: ConvBnSilu::new(c, c, 3, 1, 1),
            cv2: ConvBnSilu::new(c, c, 3, 1, 1),
            shortcut,
        }
    }

    fn call(&self, x: SymTensor) -> SymTensor {
        let y = self.cv2.call(self.cv1.call(x.clone()));
        if self.shortcut { add(x, y) } else { y }
    }
}

/// C2f block: cross-stage partial with n bottleneck layers.
///
/// Input:  `[B, in_ch, H, W]`
/// Output: `[B, out_ch, H, W]`
struct C2f<D: Float> {
    cv1:         ConvBnSilu<D>,
    cv2:         ConvBnSilu<D>,
    bottlenecks: Vec<Bottleneck<D>>,
    c_hidden:    usize,
    n:           usize,
}

impl<D: Float + 'static> C2f<D> {
    /// `n` = number of bottleneck layers.
    fn new(in_ch: usize, out_ch: usize, n: usize, shortcut: bool) -> Self {
        let c_hidden = out_ch / 2;
        let bottlenecks = (0..n).map(|_| Bottleneck::new(c_hidden, shortcut)).collect();
        // cv1: reduces to 2*c_hidden; cv2: fuses (2 + n) * c_hidden → out_ch
        Self {
            cv1: ConvBnSilu::new(in_ch, 2 * c_hidden, 1, 1, 0),
            cv2: ConvBnSilu::new((2 + n) * c_hidden, out_ch, 1, 1, 0),
            bottlenecks,
            c_hidden,
            n,
        }
    }

    fn call(&self, x: SymTensor) -> SymTensor {
        let c = self.c_hidden;
        let c2 = 2 * c;

        let y = self.cv1.call(x);
        // Split into [B, c, H, W] × 2
        let y0 = channel_chunk(&y, c2, c, 0);
        let y1 = channel_chunk(&y, c2, c, c);

        let mut parts = vec![y0, y1.clone()];
        let mut cur = y1;
        for bn in &self.bottlenecks {
            cur = bn.call(cur);
            parts.push(cur.clone());
        }

        let c_total = (2 + self.n) * c;
        let cat = channel_cat(&parts, c_total);
        self.cv2.call(cat)
    }
}

// ── MultiScaleProjector ───────────────────────────────────────────────────────

/// Three-scale feature projector for RF-DETR.
///
/// Parameters (auto-allocated by teenygrad runtime):
/// - per-scale ConvBnSilu (conv weight + BN weight/bias/running_mean/var)
/// - per-scale C2f (cv1, cv2, bottleneck conv weights + BN params)
///
/// Input:  `[B, in_ch, H_p, W_p]`
/// Output: `([B, out_ch, 2*H_p, 2*W_p], [B, out_ch, H_p, W_p], [B, out_ch, H_p/2, W_p/2])`
pub struct MultiScaleProjector<D: Float> {
    // Scale projections (1×1 conv + BN + SiLU)
    proj_s3: ConvBnSilu<D>,
    proj_s4: ConvBnSilu<D>,
    proj_s5: ConvBnSilu<D>,
    // Pooling for s5
    pool:    MaxPool2d<D, SymTensor, SymTensor, 4>,
    // C2f blocks per scale
    c2f_s3:  C2f<D>,
    c2f_s4:  C2f<D>,
    c2f_s5:  C2f<D>,
}

impl<D: Float + 'static> MultiScaleProjector<D> {
    /// `in_ch`  — backbone embed_dim (e.g. 768 for ViT-B)
    /// `out_ch` — output channels per scale (e.g. 256)
    /// `n_c2f`  — number of bottleneck layers in each C2f block (typically 1 or 2)
    pub fn new(in_ch: usize, out_ch: usize, n_c2f: usize) -> Self {
        Self {
            proj_s3: ConvBnSilu::new(in_ch, out_ch, 1, 1, 0),
            proj_s4: ConvBnSilu::new(in_ch, out_ch, 1, 1, 0),
            proj_s5: ConvBnSilu::new(in_ch, out_ch, 1, 1, 0),
            pool:    MaxPool2d::new((2, 2), (2, 2)),
            c2f_s3:  C2f::new(out_ch, out_ch, n_c2f, false),
            c2f_s4:  C2f::new(out_ch, out_ch, n_c2f, false),
            c2f_s5:  C2f::new(out_ch, out_ch, n_c2f, false),
        }
    }

    /// Forward pass — returns `(s3, s4, s5)`.
    pub fn call(&self, x: SymTensor) -> (SymTensor, SymTensor, SymTensor) {
        // ── s3: upsample × 2 ──────────────────────────────────────────────────
        let s3 = {
            let _g = name_scope("s3");
            let up  = upsample_2x(x.clone());
            let p   = self.proj_s3.call(up);
            self.c2f_s3.call(p)
        };

        // ── s4: identity ───────────────────────────────────────────────────────
        let s4 = {
            let _g = name_scope("s4");
            let p  = self.proj_s4.call(x.clone());
            self.c2f_s4.call(p)
        };

        // ── s5: maxpool × 2 ───────────────────────────────────────────────────
        let s5 = {
            let _g = name_scope("s5");
            let dn = self.pool.call(x);
            let p  = self.proj_s5.call(dn);
            self.c2f_s5.call(p)
        };

        (s3, s4, s5)
    }
}
