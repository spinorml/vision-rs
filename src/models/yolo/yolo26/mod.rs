/*
 * SpinorML Ltd 🚀 AGPL-3.0 License - https://spinorml.com/license
 */

//! YOLO26 detection model.
//!
//! Architecture reference: `ultralytics/cfg/models/26/yolo26.yaml`.
//!
//! Returns raw `DetectOutput { boxes (B, 4·A), scores (B, nc·A) }` — the
//! training-mode layout.  For inference, apply `detect_decode` on `boxes`
//! using the anchor grid and strides [8, 16, 32] appropriate for the
//! runtime input resolution.

use teeny_core::{dtype::Float, graph::SymTensor, name_scope::name_scope};

use blocks::{
    c2psa::c2psa,
    c3k2::{c3k2, c3k2_psa},
    concat::concat,
    conv::conv,
    detect::{DetectOutput, detect},
    sppf::sppf,
    upsample::upsample,
};

pub mod blocks;

// ── Variant + Config ──────────────────────────────────────────────────────────

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
    pub fn config(&self) -> Yolo26Config {
        match self {
            Yolo26Variant::N => Yolo26Config {
                depth: 0.5,
                width: 0.25,
                mc: 1024,
            },
            Yolo26Variant::S => Yolo26Config {
                depth: 0.5,
                width: 0.50,
                mc: 1024,
            },
            Yolo26Variant::M => Yolo26Config {
                depth: 0.5,
                width: 1.00,
                mc: 512,
            },
            Yolo26Variant::L => Yolo26Config {
                depth: 1.0,
                width: 1.00,
                mc: 512,
            },
            Yolo26Variant::XL => Yolo26Config {
                depth: 1.0,
                width: 1.50,
                mc: 512,
            },
        }
    }
}

// ── Scaling helpers ───────────────────────────────────────────────────────────

/// Scale a yaml base channel count by width, capped at max_channels.
fn ch(base: usize, width: f32, mc: usize) -> usize {
    (base.min(mc) as f32 * width).round() as usize
}

/// Scale a yaml repeat count by depth, minimum 1.
fn rep(base: usize, depth: f32) -> usize {
    if base > 1 {
        ((base as f32 * depth).max(1.0)).round() as usize
    } else {
        base
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// YOLO26 detection model forward closure.
///
/// All block functions are constructed eagerly and captured by the returned
/// closure.  Skip connections are implemented as plain `let` bindings inside
/// the closure body — no `Box<dyn Fn>` or shared context struct needed.
///
/// # Arguments
/// * `nc`      — number of detection classes (e.g. 80 for COCO)
/// * `variant` — model size variant (N / S / M / L / XL)
///
/// # Channels per variant (c0/c1/c2/c3/c4)
/// | N  | 16  / 32  / 64  / 128 / 256 |
/// | S  | 32  / 64  / 128 / 256 / 512 |
/// | M  | 64  / 128 / 256 / 512 / 512 |
/// | L  | 64  / 128 / 256 / 512 / 512 |
/// | XL | 96  / 192 / 384 / 512 / 512 |
pub fn yolo26<D: Float + 'static>(
    nc: usize,
    variant: &Yolo26Variant,
) -> impl Fn(SymTensor) -> DetectOutput {
    let cfg = variant.config();
    let (d, w, mc) = (cfg.depth, cfg.width, cfg.mc);

    // Scaled channel widths (yaml base → actual)
    let c0 = ch(64, w, mc); // stem output  (P1/2)
    let c1 = ch(128, w, mc); // P2/4
    let c2 = ch(256, w, mc); // P3/8   — detect small
    let c3 = ch(512, w, mc); // P4/16  — detect medium
    let c4 = ch(1024, w, mc); // P5/32  — detect large (capped by mc for M/L/XL)

    // Scaled repeat counts
    let n = rep(2, d); // n=1 for N/S/M, n=2 for L/XL

    // ── Backbone ──────────────────────────────────────────────────────────────
    //
    // Layer numbering matches the yaml (0-indexed):
    //   0: Conv  3→c0  3×3 s=2
    //   1: Conv  c0→c1 3×3 s=2
    //   2: C3k2  c1→c2 (c3k=false, e=0.25)
    //   3: Conv  c2→c2 3×3 s=2
    //   4: C3k2  c2→c3 (c3k=false, e=0.25)   ← P3 skip
    //   5: Conv  c3→c3 3×3 s=2
    //   6: C3k2  c3→c3 (c3k=true)             ← P4 skip
    //   7: Conv  c3→c4 3×3 s=2
    //   8: C3k2  c4→c4 (c3k=true)
    //   9: SPPF  c4→c4
    //  10: C2PSA c4→c4                         ← P5 skip

    let l0 = conv::<D>(3, c0, 3, 2);
    let l1 = conv::<D>(c0, c1, 3, 2);
    let l2 = c3k2::<D>(c1, c2, n, false, true, 0.25);
    let l3 = conv::<D>(c2, c2, 3, 2);
    let l4 = c3k2::<D>(c2, c3, n, false, true, 0.25); // → p3
    let l5 = conv::<D>(c3, c3, 3, 2);
    let l6 = c3k2::<D>(c3, c3, n, true, true, 0.5); // → p4
    let l7 = conv::<D>(c3, c4, 3, 2);
    let l8 = c3k2::<D>(c4, c4, n, true, true, 0.5);
    let l9 = sppf::<D>(c4, c4, true);
    let l10 = c2psa::<D>(c4, c4, n, 0.5); // → p5

    // ── Head ──────────────────────────────────────────────────────────────────
    //
    // Top-down (upsample) path:
    //  11: Upsample ×2
    //  12: Concat [11, 6=p4]       c4+c3
    //  13: C3k2  c4+c3 → c3       ← neck4 skip
    //
    //  14: Upsample ×2
    //  15: Concat [14, 4=p3]       c3+c3
    //  16: C3k2  c3+c3 → c2       → p3_det (P3/8 small)
    //
    // Bottom-up (downsample) path:
    //  17: Conv  c2→c2 3×3 s=2
    //  18: Concat [17, 13=neck4]   c2+c3
    //  19: C3k2  c2+c3 → c3       → p4_det (P4/16 medium)
    //
    //  20: Conv  c3→c3 3×3 s=2
    //  21: Concat [20, 10=p5]      c3+c4
    //  22: C3k2  c3+c4 → c4  n=1  → p5_det (P5/32 large)
    //
    //  23: Detect([p3_det, p4_det, p5_det], nc)

    let up = upsample(2, 2);
    let cat = concat();

    let l13 = c3k2::<D>(c4 + c3, c3, n, true, true, 0.5); // neck4
    let l16 = c3k2::<D>(c3 + c3, c2, n, true, true, 0.5); // p3_det
    let l17 = conv::<D>(c2, c2, 3, 2);
    let l19 = c3k2::<D>(c2 + c3, c3, n, true, true, 0.5); // p4_det
    let l20 = conv::<D>(c3, c3, 3, 2);
    // Layer 22 uses Sequential([Bottleneck, PSABlock]) as its inner block (n=1).
    let l22 = c3k2_psa::<D>(c3 + c4, c4, 1, true, 0.5); // p5_det

    let head = detect::<D>(nc, &[c2, c3, c4]);

    // ── Forward ───────────────────────────────────────────────────────────────

    move |x: SymTensor| {
        // Backbone — layer names match ultralytics yaml (0-indexed)
        let x = {
            let _g = name_scope("model.0");
            l0(x)
        };
        let x = {
            let _g = name_scope("model.1");
            l1(x)
        };
        let x = {
            let _g = name_scope("model.2");
            l2(x)
        };
        let x = {
            let _g = name_scope("model.3");
            l3(x)
        };
        let p3 = {
            let _g = name_scope("model.4");
            l4(x)
        }; // P3/8 skip
        let x = {
            let _g = name_scope("model.5");
            l5(p3.clone())
        };
        let p4 = {
            let _g = name_scope("model.6");
            l6(x)
        }; // P4/16 skip
        let x = {
            let _g = name_scope("model.7");
            l7(p4.clone())
        };
        let x = {
            let _g = name_scope("model.8");
            l8(x)
        };
        let x = {
            let _g = name_scope("model.9");
            l9(x)
        };
        let p5 = {
            let _g = name_scope("model.10");
            l10(x)
        }; // P5/32 skip

        // Top-down neck
        let x = up(p5.clone());
        let x = cat(vec![x, p4]);
        let nk4 = {
            let _g = name_scope("model.13");
            l13(x)
        }; // neck4 skip

        let x = up(nk4.clone());
        let x = cat(vec![x, p3]);
        let p3d = {
            let _g = name_scope("model.16");
            l16(x)
        }; // P3/8 det

        // Bottom-up path
        let x = {
            let _g = name_scope("model.17");
            l17(p3d.clone())
        };
        let x = cat(vec![x, nk4]);
        let p4d = {
            let _g = name_scope("model.19");
            l19(x)
        }; // P4/16 det

        let x = {
            let _g = name_scope("model.20");
            l20(p4d.clone())
        };
        let x = cat(vec![x, p5]);
        let p5d = {
            let _g = name_scope("model.22");
            l22(x)
        }; // P5/32 det

        {
            let _g = name_scope("model.23");
            head(vec![p3d, p4d, p5d])
        }
    }
}
