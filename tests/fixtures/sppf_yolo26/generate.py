#!/usr/bin/env python3
"""Generate fixtures for the SPPF CUDA forward test.

Parameters: SPPF(c_in=32, c_out=32, k=5)

    c = c_in // 2 = 16
    cv1:  Conv(32 → 16, 1×1, s=1, p=0) → BN          (act=False, no SiLU)
    pool: MaxPool2d(kernel=5, stride=1, padding=2)  × 3
    cv2:  Conv(4×16=64 → 32, 1×1, s=1, p=0) → BN → SiLU

    Forward:
        y  = cv1(x)                             # (B, 16, H, W)
        p1 = pool(y)                            # (B, 16, H, W)
        p2 = pool(p1)                           # (B, 16, H, W)
        p3 = pool(p2)                           # (B, 16, H, W)
        return cv2(cat([y, p1, p2, p3], dim=1)) # (B, 32, H, W)

    MaxPool2d with k=5, stride=1, padding=2 gives same-spatial output.

Files written (all flat little-endian f32):
  x.bin               — input  (2, 32, 4, 4) NCHW
  cv1_conv_w.bin      — cv1 Conv2d weight (16, 32, 1, 1)
  cv1_bn_w.bin        — cv1 BN γ  (16,)
  cv1_bn_b.bin        — cv1 BN β  (16,)
  cv1_bn_rm.bin       — cv1 BN running_mean (16,)
  cv1_bn_rv.bin       — cv1 BN running_var  (16,)
  cv2_conv_w.bin      — cv2 Conv2d weight (32, 64, 1, 1)
  cv2_bn_w.bin        — cv2 BN γ  (32,)
  cv2_bn_b.bin        — cv2 BN β  (32,)
  cv2_bn_rm.bin       — cv2 BN running_mean (32,)
  cv2_bn_rv.bin       — cv2 BN running_var  (32,)
  expected_output.bin — forward output (2, 32, 4, 4) NCHW  [eval mode]

Usage:
    pip install ultralytics
    python tests/fixtures/sppf_yolo26/generate.py
"""

import os
import numpy as np
import torch
from ultralytics.nn.modules.block import SPPF

torch.manual_seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))

B, C_IN, C_OUT, H, W = 2, 32, 32, 4, 4


def save(name, tensor):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    arr.tofile(os.path.join(BASE, name))
    print(f"  {name:55s}  {list(arr.shape)}")


def save_conv_block(prefix, blk):
    """Save all parameters for one ultralytics Conv block (conv + bn)."""
    save(f"{prefix}_conv_w.bin", blk.conv.weight)
    save(f"{prefix}_bn_w.bin",   blk.bn.weight)
    save(f"{prefix}_bn_b.bin",   blk.bn.bias)
    save(f"{prefix}_bn_rm.bin",  blk.bn.running_mean)
    save(f"{prefix}_bn_rv.bin",  blk.bn.running_var)


# ── Model ─────────────────────────────────────────────────────────────────────
model = SPPF(C_IN, C_OUT, k=5)
model.eval()

# ── Input + reference output ──────────────────────────────────────────────────
x = torch.randn(B, C_IN, H, W)
with torch.no_grad():
    y = model(x)

save("x.bin",               x)
save("expected_output.bin", y)

# ── Weights ───────────────────────────────────────────────────────────────────
print("\ncv1")
save_conv_block("cv1", model.cv1)
print("\ncv2")
save_conv_block("cv2", model.cv2)

c = C_IN // 2
print(f"\nModel summary:")
print(f"  c_in={C_IN}, c_out={C_OUT}, c={c}")
print(f"  cv1: Conv({C_IN} → {c}, 1×1)   weight {list(model.cv1.conv.weight.shape)}")
print(f"  pool: MaxPool2d(k=5, stride=1, padding=2)")
print(f"  cv2: Conv({4*c} → {C_OUT}, 1×1)  weight {list(model.cv2.conv.weight.shape)}")
print(f"  input: ({B}, {C_IN}, {H}, {W})")
print(f"  output: {list(y.shape)}")
print(f"\ndone — fixtures written to {BASE}")
