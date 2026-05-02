#!/usr/bin/env python3
"""Generate fixtures for the C2PSA CUDA forward-pass test.

C2PSA(c1=256, c2=256, n=1, e=0.5)  —  BATCH=2, H=4, W=4.

Architecture:
  c          = int(256 * 0.5) = 128   (hidden channels)
  num_heads  = c // 64 = 2
  head_dim   = c // num_heads = 64
  key_dim    = int(head_dim * 0.5) = 32
  qkv_h      = c + key_dim * num_heads * 2 = 256

  cv1:   Conv(256 → 256, 1×1) → BN(256) → SiLU
         split → a=(B,128,H,W)  b=(B,128,H,W)

  PSABlock on b:
    Attention:
      qkv:  Conv(128 → 256, 1×1) → BN(256)  [no SiLU, act=False]
      Reshape → Q(B,2,32,N)  K(B,2,32,N)  V(B,2,64,N)
      attn  = softmax(Q.T @ K * scale)
      out   = (V @ attn.T).view(B,128,H,W) + pe(V_spatial)
      pe:   Conv(128 → 128, 3×3, g=128) → BN(128)  [depthwise, no SiLU]
      proj: Conv(128 → 128, 1×1) → BN(128)  [no SiLU]
      b     = b + proj_out                     (shortcut)
    FFN:
      ffn0: Conv(128 → 256, 1×1) → BN(256) → SiLU
      ffn1: Conv(256 → 128, 1×1) → BN(128)  [no SiLU]
      b     = b + ffn1_out                     (shortcut)

  cat(a, b) → (B, 256, H, W)
  cv2:   Conv(256 → 256, 1×1) → BN(256) → SiLU

All tensors saved as row-major float32.
Spatial tensors (x, expected_output, conv weights) are NCHW.
BN parameter vectors (weight, bias, running_mean, running_var) are 1-D (C,).
"""

import os
import numpy as np
import torch
from ultralytics.nn.modules.block import C2PSA

torch.manual_seed(42)

BASE  = os.path.dirname(os.path.abspath(__file__))
BATCH, H, W = 2, 4, 4
C1 = 256


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
    """Save Conv2d weight and all four BN parameters."""
    save(f"{prefix}_conv_w.bin", blk.conv.weight,        base)
    save(f"{prefix}_bn_w.bin",   blk.bn.weight,          base)
    save(f"{prefix}_bn_b.bin",   blk.bn.bias,            base)
    save(f"{prefix}_bn_rm.bin",  blk.bn.running_mean,    base)
    save(f"{prefix}_bn_rv.bin",  blk.bn.running_var,     base)


# ── Generate ──────────────────────────────────────────────────────────────────

torch.manual_seed(42)
model = C2PSA(C1, C1, n=1, e=0.5)
model.eval()

x = torch.randn(BATCH, C1, H, W)
with torch.no_grad():
    y = model(x)

print("C2PSA(256, 256, n=1, e=0.5)  BATCH=2  H=4  W=4")
print()

save("x.bin",               x, BASE)
save("expected_output.bin", y, BASE)

print()

# Outer Conv blocks (standard Conv+BN+SiLU)
save_conv_block("cv1", model.cv1, BASE)
save_conv_block("cv2", model.cv2, BASE)

print()

# PSABlock[0] — Attention sub-layers (all act=False → Conv+BN, no SiLU)
attn = model.m[0].attn
save_conv_block("m0_attn_qkv",  attn.qkv,  BASE)   # 128 → 256, 1×1
save_conv_block("m0_attn_proj", attn.proj, BASE)   # 128 → 128, 1×1
save_conv_block("m0_attn_pe",   attn.pe,   BASE)   # 128 → 128, 3×3, g=128 (depthwise)

print()

# PSABlock[0] — FFN sub-layers
save_conv_block("m0_ffn0", model.m[0].ffn[0], BASE)   # 128 → 256, 1×1, SiLU
save_conv_block("m0_ffn1", model.m[0].ffn[1], BASE)   # 256 → 128, 1×1, no SiLU

print()
print("Attention parameters:")
print(f"  num_heads = {attn.num_heads}")
print(f"  head_dim  = {attn.head_dim}")
print(f"  key_dim   = {attn.key_dim}")
print(f"  scale     = {attn.scale:.18f}")
print()
print("Done — all fixtures written to", BASE)
