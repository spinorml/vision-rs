/*
 * SpinorML Ltd 🚀 AGPL-3.0 License - https://spinorml.com/license
 */

use teeny_core::graph::SymTensor;

use crate::models::yolo::yolo26::Yolo26Config;

pub fn c3k2(_config: &Yolo26Config) -> impl Fn(SymTensor) -> SymTensor {
    let _ = _config;
    |x| x
}
