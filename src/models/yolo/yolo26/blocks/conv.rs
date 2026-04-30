/*
 * SpinorML Ltd 🚀 AGPL-3.0 License - https://spinorml.com/license
 */

use teeny_core::{
    dtype::Float,
    graph::SymTensor,
    nn::{Layer, activation::sigmoid::Silu, batchnorm::BatchNorm2d, conv2d::Conv2d},
    sequential,
};

/// Conv2d → BatchNorm2d → SiLU — the basic building block of YOLO26.
///
/// Matches `ultralytics.nn.modules.conv.Conv(c1, c2, k, s)`:
///   - bias=False (BN subsumes the bias term)
///   - autopad: p = k / 2 (same-padding for odd kernels)
pub fn conv<D: Float>(
    c_in: usize,
    c_out: usize,
    k: usize,
    s: usize,
) -> impl Fn(SymTensor) -> SymTensor {
    let p = k / 2;
    sequential![
        Conv2d::<D, SymTensor, SymTensor, 4>::new(c_in, c_out, (k, k), (s, s), (p, p), false),
        BatchNorm2d::<D, _, _, 4>::new(c_out),
        Silu::<D, _, 4>::new()
    ]
}
