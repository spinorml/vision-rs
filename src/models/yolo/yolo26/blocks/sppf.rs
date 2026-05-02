/*
 * SpinorML Ltd 🚀 AGPL-3.0 License - https://spinorml.com/license
 */

use teeny_core::{
    dtype::Float,
    graph::SymTensor,
    nn::{Layer, pool::MaxPool2d},
};

use super::{concat::concat, conv::{conv, conv_bn}};

/// Spatial Pyramid Pooling - Fast (SPPF).
///
/// Matches `ultralytics.nn.modules.block.SPPF(c1, c2, k=5)`:
///   cv1: Conv(c1, c//2, 1, 1, act=False) — Conv+BN only, no SiLU
///   pool: MaxPool2d(k=5, stride=1, padding=2) applied 3 times
///   cv2: Conv(4 * c//2, c2, 1, 1) — Conv+BN+SiLU
pub fn sppf<D: Float>(c_in: usize, c_out: usize) -> impl Fn(SymTensor) -> SymTensor {
    let c = c_in / 2;
    let cv1 = conv_bn::<D>(c_in, c, 1, 1, 1);
    let cv2 = conv::<D>(4 * c, c_out, 1, 1);
    let pool = || {
        MaxPool2d::<D, SymTensor, SymTensor, 4>::with_padding((5, 5), (1, 1), (2, 2))
    };

    move |x| {
        let y = cv1(x);
        let p1 = pool().call(y.clone());
        let p2 = pool().call(p1.clone());
        let p3 = pool().call(p2.clone());
        let cat = concat()(vec![y, p1, p2, p3]);
        cv2(cat)
    }
}
