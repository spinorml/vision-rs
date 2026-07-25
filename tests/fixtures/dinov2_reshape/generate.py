#!/usr/bin/env python3
"""Generate fixtures for dinov2_nchw_to_nld and dinov2_nld_to_nchw kernel tests.

nchw_to_nld: [B, D, H, W] → [B, N, D]  where N = H * W
nld_to_nchw: [B, N, D]   → [B, D, H, W]

The two kernels are transposes of each other:
  nld_to_nchw is also the backward of nchw_to_nld.
  nchw_to_nld is also the backward of nld_to_nchw.

Files written (flat little-endian f32):
  nchw.bin               [B, D, H, W]
  expected_nld.bin       [B, N, D]
  grad_nld.bin           [B, N, D]   — synthetic upstream gradient
  expected_dnchw.bin     [B, D, H, W] — gradient w.r.t. nchw input

  nld.bin                [B, N, D]
  expected_nchw.bin      [B, D, H, W]
  grad_nchw.bin          [B, D, H, W] — synthetic upstream gradient
  expected_dnld.bin      [B, N, D]   — gradient w.r.t. nld input

Usage:
  python tests/fixtures/dinov2_reshape/generate.py
"""

import os
import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))

B = 2    # batch size
D = 64   # embed_dim (DINOv2 ViT-S style)
H = 4    # patch rows
W = 4    # patch cols
N = H * W


def save(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    path = os.path.join(BASE, name)
    arr.tofile(path)
    print(f"  {name:40s}  shape={list(arr.shape)}")


# ── nchw_to_nld forward + backward ───────────────────────────────────────────

nchw = torch.randn(B, D, H, W)

# Forward reference: [B, D, H, W] → [B, H, W, D] → [B, N, D]
nld = nchw.permute(0, 2, 3, 1).contiguous().view(B, N, D)

# Backward: grad [B, N, D] → [B, D, H, W]
grad_nld = torch.randn(B, N, D)
dnchw = grad_nld.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

save("nchw.bin",           nchw.numpy())
save("expected_nld.bin",   nld.numpy())
save("grad_nld.bin",       grad_nld.numpy())
save("expected_dnchw.bin", dnchw.numpy())

# ── nld_to_nchw forward + backward ───────────────────────────────────────────

nld2 = torch.randn(B, N, D)

# Forward reference: [B, N, D] → [B, H, W, D] → [B, D, H, W]
nchw2 = nld2.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

# Backward: grad [B, D, H, W] → [B, N, D]
grad_nchw = torch.randn(B, D, H, W)
dnld = grad_nchw.permute(0, 2, 3, 1).contiguous().view(B, N, D)

save("nld.bin",            nld2.numpy())
save("expected_nchw.bin",  nchw2.numpy())
save("grad_nchw.bin",      grad_nchw.numpy())
save("expected_dnld.bin",  dnld.numpy())

print(f"\n  B={B}  D={D}  H={H}  W={W}  N={N}")
print(f"  nld max_abs={nld.abs().max():.6f}")
print(f"  nchw2 max_abs={nchw2.abs().max():.6f}")
print("done")
