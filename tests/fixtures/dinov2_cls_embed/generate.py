#!/usr/bin/env python3
"""Generate fixtures for dinov2 cls-embed kernel tests.

cat_cls:        [B, N, D] + cls[D]    → [B, N+1, D]
add_pos_embed:  [B, N, D] + pos[N, D] → [B, N, D]
remove_cls:     [B, N+1, D]           → [B, N, D]

Files written (flat little-endian f32):
  cat_cls/tokens.bin             [B, N, D]
  cat_cls/cls.bin                [D]
  cat_cls/expected_out.bin       [B, N+1, D]
  cat_cls/grad_out.bin           [B, N+1, D]
  cat_cls/expected_dtokens.bin   [B, N, D]
  cat_cls/expected_dcls.bin      [D]

  add_pos/tokens.bin             [B, N, D]
  add_pos/pos.bin                [N, D]
  add_pos/expected_out.bin       [B, N, D]
  add_pos/grad_out.bin           [B, N, D]
  add_pos/expected_dtokens.bin   [B, N, D]
  add_pos/expected_dpos.bin      [N, D]

  remove_cls/tokens.bin          [B, N+1, D]
  remove_cls/expected_out.bin    [B, N, D]
  remove_cls/grad_out.bin        [B, N, D]
  remove_cls/expected_din.bin    [B, N+1, D]   (cls slot is zero)

Usage:
  python tests/fixtures/dinov2_cls_embed/generate.py
"""

import os
import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))

B = 2    # batch
N = 16   # patch tokens
D = 64   # embed dim


def save(subdir, name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    d = os.path.join(BASE, subdir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    arr.tofile(path)
    print(f"  {subdir}/{name:30s}  shape={list(arr.shape)}")


# ── cat_cls ───────────────────────────────────────────────────────────────────

tokens = torch.randn(B, N, D)
cls    = torch.randn(D)

# Forward: prepend cls to each batch
cls_broadcast = cls.unsqueeze(0).unsqueeze(0).expand(B, 1, D)   # [B, 1, D]
out_cat = torch.cat([cls_broadcast, tokens], dim=1)             # [B, N+1, D]

# Backward
grad_cat = torch.randn(B, N + 1, D)
d_tokens = grad_cat[:, 1:, :]               # [B, N, D]
d_cls    = grad_cat[:, 0, :].sum(dim=0)     # [D] (sum over batch)

save("cat_cls", "tokens.bin",           tokens.numpy())
save("cat_cls", "cls.bin",              cls.numpy())
save("cat_cls", "expected_out.bin",     out_cat.numpy())
save("cat_cls", "grad_out.bin",         grad_cat.numpy())
save("cat_cls", "expected_dtokens.bin", d_tokens.numpy())
save("cat_cls", "expected_dcls.bin",    d_cls.numpy())

# ── add_pos_embed ─────────────────────────────────────────────────────────────

tokens_p = torch.randn(B, N, D)
pos      = torch.randn(N, D)

# Forward: broadcast add
out_pos = tokens_p + pos.unsqueeze(0)       # [B, N, D]

# Backward
grad_pos = torch.randn(B, N, D)
d_tokens_p = grad_pos.clone()
d_pos      = grad_pos.sum(dim=0)            # [N, D] (sum over batch)

save("add_pos", "tokens.bin",           tokens_p.numpy())
save("add_pos", "pos.bin",              pos.numpy())
save("add_pos", "expected_out.bin",     out_pos.numpy())
save("add_pos", "grad_out.bin",         grad_pos.numpy())
save("add_pos", "expected_dtokens.bin", d_tokens_p.numpy())
save("add_pos", "expected_dpos.bin",    d_pos.numpy())

# ── remove_cls ────────────────────────────────────────────────────────────────

tokens_r = torch.randn(B, N + 1, D)

# Forward: slice off position 0
out_r = tokens_r[:, 1:, :]                 # [B, N, D]

# Backward: gradient of the N patch positions; cls gets 0
grad_r = torch.randn(B, N, D)
d_in   = torch.zeros(B, N + 1, D)
d_in[:, 1:, :] = grad_r

save("remove_cls", "tokens.bin",       tokens_r.numpy())
save("remove_cls", "expected_out.bin", out_r.numpy())
save("remove_cls", "grad_out.bin",     grad_r.numpy())
save("remove_cls", "expected_din.bin", d_in.numpy())

print(f"\n  B={B}  N={N}  D={D}")
print("done")
