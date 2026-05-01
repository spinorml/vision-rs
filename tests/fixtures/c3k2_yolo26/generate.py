#!/usr/bin/env python3
"""Generate fixtures for the C3k2 CUDA forward test.

Parameters (YOLO11n-n backbone layer 2, without shortcut for simplicity):
    C3k2(c_in=32, c_out=64, n=1, c3k=False, shortcut=False, e=0.25)

    c = int(64 * 0.25) = 16
    c_inner = int(16 * 0.5) = 8  ← Bottleneck default e=0.5
    cv1:     Conv(32 → 32, 1×1)      [c_in → 2c]
    m[0].cv1 Conv(16 → 8,  3×3)      [c → c_inner, k=3 default]
    m[0].cv2 Conv(8  → 16, 3×3)      [c_inner → c, k=3 default]
    cv2:     Conv(48 → 64, 1×1)      [(2+n)c → c_out]

    Forward (C2f base):
        y = list(cv1(x).chunk(2, dim=1))   # [y0, y1], each (B, 16, H, W)
        y.extend(m(y[-1]) for m in self.m) # [y0, y1, m0(y1)]
        return cv2(cat(y, dim=1))          # (B, 64, H, W)

Layout note:
    PyTorch stores tensors in NCHW. All fixture files are saved in NCHW
    row-major order. The Rust test converts to NC (channels-last) as needed
    before each BN / chunk / cat kernel launch.

Usage:
    pip install ultralytics
    python generate.py
"""

import os
import numpy as np
import torch
from ultralytics.nn.modules.block import C3k2

torch.manual_seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))


def save(name, tensor):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    arr.tofile(os.path.join(BASE, name))
    print(f"  {name:55s}  {list(arr.shape)}")


def save_conv_block(prefix, blk):
    """Save all parameters for one ultralytics Conv block (conv + bn)."""
    save(f"{prefix}_conv_w.bin",  blk.conv.weight)
    save(f"{prefix}_bn_w.bin",    blk.bn.weight)
    save(f"{prefix}_bn_b.bin",    blk.bn.bias)
    save(f"{prefix}_bn_rm.bin",   blk.bn.running_mean)
    save(f"{prefix}_bn_rv.bin",   blk.bn.running_var)


# ── Model ─────────────────────────────────────────────────────────────────────
B, C_IN, C_OUT, H, W = 2, 32, 64, 4, 4

model = C3k2(C_IN, C_OUT, n=1, c3k=False, shortcut=False, e=0.25)
model.eval()

# ── Input + reference output ──────────────────────────────────────────────────
x = torch.randn(B, C_IN, H, W)
with torch.no_grad():
    y = model(x)

save("x.bin",               x)
save("expected_output.bin", y)

# ── Weights ───────────────────────────────────────────────────────────────────
print("\ncv1")
save_conv_block("cv1",    model.cv1)
print("\nm[0].cv1")
save_conv_block("m0_cv1", model.m[0].cv1)
print("\nm[0].cv2")
save_conv_block("m0_cv2", model.m[0].cv2)
print("\ncv2")
save_conv_block("cv2",    model.cv2)

print("\ndone — fixtures written to", BASE)
