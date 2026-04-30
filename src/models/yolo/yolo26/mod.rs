/*
 * SpinorML Ltd 🚀 AGPL-3.0 License - https://spinorml.com/license
 */

use teeny_core::graph::SymTensor;

pub mod blocks;

#[derive(Debug)]
pub struct Yolo26Config {
    pub depth: f32,
    pub width: f32,
    pub mc: usize,
}

#[derive(Debug)]
pub enum Yolo26Variant {
    N,
    S,
    M,
    L,
    XL,
}

impl Yolo26Variant {
    pub fn depth(&self) -> Yolo26Config {
        match self {
            Yolo26Variant::N => Yolo26Config {
                depth: 0.5,
                width: 0.25,
                mc: 1024,
            },
            Yolo26Variant::S => Yolo26Config {
                depth: 0.5,
                width: 0.5,
                mc: 1024,
            },
            Yolo26Variant::M => Yolo26Config {
                depth: 0.5,
                width: 1.0,
                mc: 512,
            },
            Yolo26Variant::L => Yolo26Config {
                depth: 1.0,
                width: 1.0,
                mc: 512,
            },
            Yolo26Variant::XL => Yolo26Config {
                depth: 1.0,
                width: 1.5,
                mc: 512,
            },
        }
    }
}

pub fn yolo26(variant: Yolo26Variant) {
    let _backbone = backbone(&variant);
    let _neck = neck(&variant);
    let _head = head(&variant);

    todo!()
}

fn backbone(variant: &Yolo26Variant) -> impl Fn(SymTensor) -> SymTensor {
    let _ = variant;
    |x| x
}

fn neck(variant: &Yolo26Variant) -> impl Fn(SymTensor) -> SymTensor {
    let _ = variant;
    |x| x
}

fn head(variant: &Yolo26Variant) -> impl Fn(SymTensor) -> SymTensor {
    let _ = variant;
    |x| x
}
