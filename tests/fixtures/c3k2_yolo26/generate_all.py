#!/usr/bin/env python3
"""Generate fixtures for all C3k2 CUDA tests.

Variants (all YOLO11n-n layers, n=1):
  1. backbone_shallow_sf: C3k2(32,  64,  c3k=F, shortcut=F, e=0.25)  existing dir
  2. shortcut/:           C3k2(32,  64,  c3k=F, shortcut=T, e=0.25)
  3. deep_sf/:            C3k2(128, 128, c3k=T, shortcut=F, e=0.5)
  4. deep_st/:            C3k2(128, 128, c3k=T, shortcut=T, e=0.5)
  5. head_sf/:            C3k2(384, 128, c3k=F, shortcut=F, e=0.5)
  6. head_st/:            C3k2(384, 256, c3k=T, shortcut=T, e=0.5)
  7. backward/:           C3k2(32,  64,  c3k=F, shortcut=F, e=0.25), training mode
                          Saves BN intermediates (mean, rstd, pre-BN, pre-SiLU)
                          and the reference dx from PyTorch autograd.

All tensors saved as row-major float32.  Spatial tensors are NCHW.
BN parameter tensors (weight, bias, running_mean, running_var) are 1-D (C,).
NC intermediate tensors have shape (B*H*W, C) in row-major order.
"""

import os
import numpy as np
import torch
from ultralytics.nn.modules.block import C3k2

torch.manual_seed(42)
SEED = 42

BASE = os.path.dirname(os.path.abspath(__file__))
BATCH, H, W = 2, 4, 4


# ── helpers ───────────────────────────────────────────────────────────────────

def save(name, tensor, base=None):
    if base is None:
        base = BASE
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    os.makedirs(base, exist_ok=True)
    arr.tofile(os.path.join(base, name))
    print(f"  {name:60s}  {list(arr.shape)}")


def save_conv_block(prefix, blk, base):
    save(f"{prefix}_conv_w.bin",  blk.conv.weight, base)
    save(f"{prefix}_bn_w.bin",    blk.bn.weight,   base)
    save(f"{prefix}_bn_b.bin",    blk.bn.bias,      base)
    save(f"{prefix}_bn_rm.bin",   blk.bn.running_mean, base)
    save(f"{prefix}_bn_rv.bin",   blk.bn.running_var,  base)


def nchw_to_nc(x_nchw):
    """(B,C,H,W) → (B*H*W, C) row-major NC."""
    B, C, H, W = x_nchw.shape
    return x_nchw.permute(0, 2, 3, 1).contiguous().reshape(-1, C)


# ── C3k2 c3k=False variants ───────────────────────────────────────────────────

def gen_c3k_false(base, c_in, c_out, shortcut, e, regen_existing=False):
    if os.path.exists(os.path.join(base, "expected_output.bin")) and not regen_existing:
        print(f"  (skipping {base} — already exists)")
        return

    torch.manual_seed(SEED)
    model = C3k2(c_in, c_out, n=1, c3k=False, shortcut=shortcut, e=e)
    model.eval()

    x = torch.randn(BATCH, c_in, H, W)
    with torch.no_grad():
        y = model(x)

    print(f"\n[c3k=False c_in={c_in} c_out={c_out} sc={shortcut} e={e}]  base={base}")
    save("x.bin",               x,    base)
    save("expected_output.bin", y,    base)
    save_conv_block("cv1",    model.cv1,      base)
    save_conv_block("m0_cv1", model.m[0].cv1, base)
    save_conv_block("m0_cv2", model.m[0].cv2, base)
    save_conv_block("cv2",    model.cv2,      base)


# ── C3k2 c3k=True variants ────────────────────────────────────────────────────

def gen_c3k_true(base, c_in, c_out, shortcut, e):
    if os.path.exists(os.path.join(base, "expected_output.bin")):
        print(f"  (skipping {base} — already exists)")
        return

    torch.manual_seed(SEED)
    model = C3k2(c_in, c_out, n=1, c3k=True, shortcut=shortcut, e=e)
    model.eval()

    x = torch.randn(BATCH, c_in, H, W)
    with torch.no_grad():
        y = model(x)

    print(f"\n[c3k=True c_in={c_in} c_out={c_out} sc={shortcut} e={e}]  base={base}")
    save("x.bin",               x, base)
    save("expected_output.bin", y, base)

    save_conv_block("cv1",      model.cv1,             base)  # outer cv1
    # m[0] is a C3k module (inherits C3)
    save_conv_block("m0_icv1",   model.m[0].cv1,       base)  # C3k cv1 (c → c_h, 1×1)
    save_conv_block("m0_icv2",   model.m[0].cv2,       base)  # C3k cv2 (c → c_h, 1×1)
    save_conv_block("m0_icv3",   model.m[0].cv3,       base)  # C3k cv3 (2*c_h → c, 1×1)
    save_conv_block("m0_m0_cv1", model.m[0].m[0].cv1,  base)  # inner bottleneck 0, cv1
    save_conv_block("m0_m0_cv2", model.m[0].m[0].cv2,  base)  # inner bottleneck 0, cv2
    save_conv_block("m0_m1_cv1", model.m[0].m[1].cv1,  base)  # inner bottleneck 1, cv1
    save_conv_block("m0_m1_cv2", model.m[0].m[1].cv2,  base)  # inner bottleneck 1, cv2
    save_conv_block("cv2",       model.cv2,             base)  # outer cv2


# ── Backward fixtures (training mode, c3k=False shortcut=False 32→64) ─────────

def gen_backward(base):
    if os.path.exists(os.path.join(base, "dx_expected.bin")):
        print(f"  (skipping {base} — already exists)")
        return

    torch.manual_seed(SEED)
    c_in, c_out, e = 32, 64, 0.25

    model = C3k2(c_in, c_out, n=1, c3k=False, shortcut=False, e=e)
    model.train()

    # capture pre-BN input and BN stats via hooks
    intermediates = {}

    def make_conv_hook(name):
        def hook(module, inp, out):
            intermediates[f"{name}_pre_bn_nchw"] = out.detach().clone()
        return hook

    def make_bn_hook(name):
        def hook(module, inp, out):
            x_in = inp[0]                         # NCHW, pre-BN
            B, C, Hh, Ww = x_in.shape
            mean = x_in.mean(dim=(0, 2, 3))
            var  = x_in.var(dim=(0, 2, 3), unbiased=False)
            rstd = (var + 1e-5).rsqrt()
            intermediates[f"{name}_bn_mean"]     = mean.detach()
            intermediates[f"{name}_bn_rstd"]     = rstd.detach()
            intermediates[f"{name}_pre_silu_nchw"] = out.detach().clone()
        return hook

    for tag, blk in [("cv1",    model.cv1),
                     ("m0_cv1", model.m[0].cv1),
                     ("m0_cv2", model.m[0].cv2),
                     ("cv2",    model.cv2)]:
        blk.conv.register_forward_hook(make_conv_hook(tag))
        blk.bn.register_forward_hook(make_bn_hook(tag))

    x   = torch.randn(BATCH, c_in, H, W, requires_grad=True)
    y   = model(x)
    dy  = torch.ones_like(y)
    y.backward(dy)
    dx  = x.grad.detach()

    print(f"\n[backward training c_in={c_in} c_out={c_out} e={e}]  base={base}")
    save("x.bin",           x.detach(), base)
    save("dy.bin",          dy,          base)
    save("dx_expected.bin", dx,          base)
    save("expected_output.bin", y.detach(), base)

    save_conv_block("cv1",    model.cv1,      base)
    save_conv_block("m0_cv1", model.m[0].cv1, base)
    save_conv_block("m0_cv2", model.m[0].cv2, base)
    save_conv_block("cv2",    model.cv2,      base)

    # Per-stage NC intermediates  (shape B*H*W × C)
    for tag in ["cv1", "m0_cv1", "m0_cv2", "cv2"]:
        pre_bn_nchw   = intermediates[f"{tag}_pre_bn_nchw"]
        pre_silu_nchw = intermediates[f"{tag}_pre_silu_nchw"]
        mean          = intermediates[f"{tag}_bn_mean"]
        rstd          = intermediates[f"{tag}_bn_rstd"]

        pre_bn_nc   = nchw_to_nc(pre_bn_nchw)
        pre_silu_nc = nchw_to_nc(pre_silu_nchw)

        save(f"{tag}_pre_bn_nc.bin",   pre_bn_nc,   base)
        save(f"{tag}_bn_mean.bin",     mean,         base)
        save(f"{tag}_bn_rstd.bin",     rstd,         base)
        save(f"{tag}_pre_silu_nc.bin", pre_silu_nc,  base)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Existing (backbone_shallow_sf) — regenerate to sync with updated bottleneck
    gen_c3k_false(BASE,
                  c_in=32, c_out=64, shortcut=False, e=0.25,
                  regen_existing=True)

    # 2. backbone_shallow_st
    gen_c3k_false(os.path.join(BASE, "shortcut"),
                  c_in=32, c_out=64, shortcut=True, e=0.25)

    # 3. backbone_deep_sf
    gen_c3k_true(os.path.join(BASE, "deep_sf"),
                 c_in=128, c_out=128, shortcut=False, e=0.5)

    # 4. backbone_deep_st
    gen_c3k_true(os.path.join(BASE, "deep_st"),
                 c_in=128, c_out=128, shortcut=True, e=0.5)

    # 5. head_sf
    gen_c3k_false(os.path.join(BASE, "head_sf"),
                  c_in=384, c_out=128, shortcut=False, e=0.5)

    # 6. head_st
    gen_c3k_true(os.path.join(BASE, "head_st"),
                 c_in=384, c_out=256, shortcut=True, e=0.5)

    # 7. backward
    gen_backward(os.path.join(BASE, "backward"))

    print("\ndone — all fixtures written")
