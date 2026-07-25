#!/usr/bin/env python3
"""Generate fixtures for the YOLO26 Detect head cv2[0] CUDA forward test.

Parameters (matching YOLO26n Detect head at P3/8 scale, with small spatial for speed):
    nc = 80, c_in = 256, H = W = 4, B = 1
    reg_max = 1
    c2 = max(16, c_in//4, reg_max*4) = max(16, 64, 4) = 64
    c3 = max(c_in, min(nc, 100)) = max(256, 80) = 256

cv2[0] chain:
    conv1: Conv(256→64, 3×3, s=1, p=1) → BN → SiLU
    conv2: Conv(64→64,  3×3, s=1, p=1) → BN → SiLU
    conv3: nn.Conv2d(64→4, 1×1, bias=True)  — plain conv, no BN, no act

Files written (all flat little-endian f32):
    x.bin                 — input (1, 256, 4, 4) NCHW
    conv1_w.bin           — cv2[0] conv1 weight (64, 256, 3, 3)
    conv1_bn_w.bin        — cv2[0] conv1 BN gamma  (64,)
    conv1_bn_b.bin        — cv2[0] conv1 BN beta   (64,)
    conv1_bn_rm.bin       — cv2[0] conv1 BN running_mean (64,)
    conv1_bn_rv.bin       — cv2[0] conv1 BN running_var  (64,)
    conv2_w.bin           — cv2[0] conv2 weight (64, 64, 3, 3)
    conv2_bn_w.bin        — cv2[0] conv2 BN gamma  (64,)
    conv2_bn_b.bin        — cv2[0] conv2 BN beta   (64,)
    conv2_bn_rm.bin       — cv2[0] conv2 BN running_mean (64,)
    conv2_bn_rv.bin       — cv2[0] conv2 BN running_var  (64,)
    conv3_w.bin           — cv2[0] conv3 plain Conv2d weight (4, 64, 1, 1)
    conv3_bias.bin        — cv2[0] conv3 plain Conv2d bias   (4,)
    expected_output.bin   — cv2[0](x) output (1, 4, 4, 4) NCHW

Usage:
    pip install ultralytics
    python tests/fixtures/detect_yolo26/generate.py
"""

import os
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))

B, C_IN, H, W = 1, 256, 4, 4
NC = 80
REG_MAX = 1
C2 = max(16, C_IN // 4, REG_MAX * 4)  # 64
C3 = max(C_IN, min(NC, 100))           # 256

assert C2 == 64
assert C3 == 256


def save(name, tensor):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    arr.tofile(os.path.join(BASE, name))
    print(f"  {name:55s}  {list(arr.shape)}")


def save_conv_block(prefix, blk):
    """Save all parameters for one ultralytics Conv block (conv + bn)."""
    save(f"{prefix}_w.bin",    blk.conv.weight)
    save(f"{prefix}_bn_w.bin", blk.bn.weight)
    save(f"{prefix}_bn_b.bin", blk.bn.bias)
    save(f"{prefix}_bn_rm.bin", blk.bn.running_mean)
    save(f"{prefix}_bn_rv.bin", blk.bn.running_var)


# ── Build cv2[0] manually (matching ultralytics Detect head exactly) ──────────
#
# ultralytics Detect.__init__:
#   self.cv2 = nn.ModuleList(
#       nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
#                     nn.Conv2d(c2, 4 * self.reg_max, 1))
#       for x in ch)
#
# ultralytics Conv = Conv2d + BN + SiLU (bias=False, same-padding).

from ultralytics.nn.modules.conv import Conv as UltralyticsConv

conv1 = UltralyticsConv(C_IN, C2, 3)   # Conv(256→64, 3×3)
conv2 = UltralyticsConv(C2, C2, 3)     # Conv(64→64, 3×3)
conv3 = nn.Conv2d(C2, 4 * REG_MAX, 1)  # plain Conv2d(64→4, 1×1, bias=True)

cv2_0 = nn.Sequential(conv1, conv2, conv3)
cv2_0.eval()

# ── Input + reference output ──────────────────────────────────────────────────
x = torch.randn(B, C_IN, H, W)
with torch.no_grad():
    y = cv2_0(x)

assert list(y.shape) == [B, 4 * REG_MAX, H, W], f"unexpected shape: {y.shape}"

print("Saving fixtures to", BASE)
save("x.bin",               x)
save("expected_output.bin", y)

print("\ncv2[0] conv1 (Conv(256→64, 3×3) + BN + SiLU):")
save_conv_block("conv1", conv1)

print("\ncv2[0] conv2 (Conv(64→64, 3×3) + BN + SiLU):")
save_conv_block("conv2", conv2)

print("\ncv2[0] conv3 (plain Conv2d(64→4, 1×1, bias=True)):")
save("conv3_w.bin",    conv3.weight)
save("conv3_bias.bin", conv3.bias)

print(f"""
Summary:
  B={B}, C_IN={C_IN}, H={H}, W={W}, NC={NC}
  reg_max={REG_MAX}, c2={C2}, c3={C3}
  cv2[0](x): {list(x.shape)} -> {list(y.shape)}

  conv1 weight: {list(conv1.conv.weight.shape)}
  conv2 weight: {list(conv2.conv.weight.shape)}
  conv3 weight: {list(conv3.weight.shape)}, bias: {list(conv3.bias.shape)}
done — fixtures written to {BASE}
""")
