#!/usr/bin/env python3
"""Generate fixtures for dinov2_pack_qkv forward and backward CUDA tests.

Layout:
  Input:  qkv[b, n, s*H*HD + h*HD + d]  (s=0:Q, 1:K, 2:V)  → [B, N, 3*H*HD]
  Output: packed[(s*BH + bh), n, d]                         → [3*BH, N, HD]
  where bh = b * H + h

Files written (flat little-endian f32):
  qkv.bin              [B, N, 3*H*HD]
  expected_packed.bin  [3*BH, N, HD]
  grad_packed.bin      [3*BH, N, HD]
  expected_dqkv.bin    [B, N, 3*H*HD]

Usage:
  python tests/fixtures/dinov2_pack_qkv/generate.py
"""

import os
import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))

B  = 2   # batch size
H  = 4   # number of attention heads
HD = 32  # head dimension
N  = 16  # sequence length
BH = B * H


def save(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    path = os.path.join(BASE, name)
    arr.tofile(path)
    print(f"  {name:40s}  shape={list(arr.shape)}")


# ── Forward ───────────────────────────────────────────────────────────────────

# Input layout: qkv[b, n, s*H*HD + h*HD + d]
qkv_data = torch.randn(B, N, 3 * H * HD)

# Reference forward:
#   view as [B, N, 3, H, HD] → permute to [3, B, H, N, HD] → view [3*BH, N, HD]
qkv_reshaped = qkv_data.view(B, N, 3, H, HD)            # [B, N, 3, H, HD]
qkv_perm     = qkv_reshaped.permute(2, 0, 3, 1, 4)      # [3, B, H, N, HD]
packed       = qkv_perm.contiguous().view(3 * BH, N, HD) # [3*BH, N, HD]

# ── Backward ──────────────────────────────────────────────────────────────────

# Gradient flowing back from Flash Attention 2 → pack_qkv input gradient
grad_packed = torch.randn(3 * BH, N, HD)

# Backward reference: reverse permutation
#   [3*BH, N, HD] → [3, BH, N, HD] → [3, B, H, N, HD] → [B, N, 3, H, HD] → [B, N, 3*H*HD]
gp_view  = grad_packed.view(3, B, H, N, HD)              # [3, B, H, N, HD]
dqkv     = gp_view.permute(1, 3, 0, 2, 4).contiguous().view(B, N, 3 * H * HD)

# ── Save ──────────────────────────────────────────────────────────────────────

print(f"\nSaving fixtures to {BASE}")
print(f"  B={B}  H={H}  HD={HD}  N={N}  BH={BH}")
print()

save("qkv.bin",              qkv_data.detach().numpy())
save("expected_packed.bin",  packed.detach().numpy())
save("grad_packed.bin",      grad_packed.numpy())
save("expected_dqkv.bin",    dqkv.numpy())

print(f"\n  packed max_abs={packed.abs().max():.6f}")
print(f"  dqkv   max_abs={dqkv.abs().max():.6f}")
print("done")
