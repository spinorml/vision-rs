#!/usr/bin/env python3
"""Generate fixtures for dinov2_unpack_attn forward and backward CUDA tests.

Layout:
  Input:  attn[bh, n, d]         → [BH, N, HD]
  Output: out[b, n, h*HD + d]    → [B, N, H*HD]
  where bh = b * H + h

Files written (flat little-endian f32):
  attn_out.bin           [BH, N, HD]
  expected_unpacked.bin  [B, N, H*HD]
  grad_unpacked.bin      [B, N, H*HD]
  expected_dattn_out.bin [BH, N, HD]

Usage:
  python tests/fixtures/dinov2_unpack_attn/generate.py
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

# Input: attn[BH, N, HD]
attn_out = torch.randn(BH, N, HD)

# Reference forward:
#   view as [B, H, N, HD] → permute to [B, N, H, HD] → view [B, N, H*HD]
attn_bh  = attn_out.view(B, H, N, HD)                          # [B, H, N, HD]
unpacked = attn_bh.permute(0, 2, 1, 3).contiguous().view(B, N, H * HD)  # [B, N, H*HD]

# ── Backward ──────────────────────────────────────────────────────────────────

# Gradient flowing from output projection backward through unpack_attn
grad_unpacked = torch.randn(B, N, H * HD)

# Backward reference: reverse permutation
#   [B, N, H*HD] → [B, N, H, HD] → [B, H, N, HD] → [BH, N, HD]
gu_view = grad_unpacked.view(B, N, H, HD)                          # [B, N, H, HD]
dattn   = gu_view.permute(0, 2, 1, 3).contiguous().view(BH, N, HD) # [BH, N, HD]

# ── Save ──────────────────────────────────────────────────────────────────────

print(f"\nSaving fixtures to {BASE}")
print(f"  B={B}  H={H}  HD={HD}  N={N}  BH={BH}")
print()

save("attn_out.bin",           attn_out.numpy())
save("expected_unpacked.bin",  unpacked.numpy())
save("grad_unpacked.bin",      grad_unpacked.numpy())
save("expected_dattn_out.bin", dattn.numpy())

print(f"\n  unpacked max_abs={unpacked.abs().max():.6f}")
print(f"  dattn    max_abs={dattn.abs().max():.6f}")
print("done")
