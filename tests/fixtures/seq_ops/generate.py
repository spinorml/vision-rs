#!/usr/bin/env python3
"""Generate fixtures for seq_ops kernel tests.

pack_heads:  [B, S, n_heads * HD] → [BH, S, HD]
seq_cat2:    [B, Sa, D] + [B, Sb, D] → [B, Sa+Sb, D]

Files written (flat little-endian f32):
  pack_heads/inp.bin           [B, S, n_heads * HD]
  pack_heads/expected_out.bin  [BH, S, HD]
  pack_heads/grad_out.bin      [BH, S, HD]
  pack_heads/expected_dinp.bin [B, S, n_heads * HD]

  seq_cat2/a.bin               [B, Sa, D]
  seq_cat2/b.bin               [B, Sb, D]
  seq_cat2/expected_out.bin    [B, Sa+Sb, D]
  seq_cat2/grad_out.bin        [B, Sa+Sb, D]
  seq_cat2/expected_da.bin     [B, Sa, D]
  seq_cat2/expected_db.bin     [B, Sb, D]

Usage:
  python tests/fixtures/seq_ops/generate.py
"""

import os
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

BASE = os.path.dirname(os.path.abspath(__file__))

B       = 2
S       = 16
N_HEADS = 4
HD      = 32
D       = N_HEADS * HD  # 128

SA = 10
SB = 6


def save(subdir, name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    d = os.path.join(BASE, subdir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    arr.tofile(path)
    print(f"  {subdir}/{name:30s}  shape={list(arr.shape)}")


# ── pack_heads ────────────────────────────────────────────────────────────────

inp = torch.randn(B, S, D)   # [B, S, n_heads * HD]

# Forward: [B, S, n_heads, HD] → [B, n_heads, S, HD] → [B*n_heads, S, HD]
# Memory layout: inp[b, s, h*HD + d] → out[b*n_heads + h, s, d]
inp_4d = inp.view(B, S, N_HEADS, HD)            # [B, S, n_heads, HD]
out_4d = inp_4d.permute(0, 2, 1, 3).contiguous()  # [B, n_heads, S, HD]
out_ph = out_4d.view(B * N_HEADS, S, HD)         # [BH, S, HD]

# Backward: grad_out [BH, S, HD] → grad_inp [B, S, n_heads * HD]
grad_out_ph = torch.randn(B * N_HEADS, S, HD)
grad_4d = grad_out_ph.view(B, N_HEADS, S, HD)
grad_back = grad_4d.permute(0, 2, 1, 3).contiguous()  # [B, S, n_heads, HD]
d_inp = grad_back.view(B, S, D)

save("pack_heads", "inp.bin",            inp.numpy())
save("pack_heads", "expected_out.bin",   out_ph.numpy())
save("pack_heads", "grad_out.bin",       grad_out_ph.numpy())
save("pack_heads", "expected_dinp.bin",  d_inp.numpy())

# ── seq_cat2 ──────────────────────────────────────────────────────────────────

a = torch.randn(B, SA, D)
b = torch.randn(B, SB, D)

out_cat = torch.cat([a, b], dim=1)  # [B, SA+SB, D]

grad_out_cat = torch.randn(B, SA + SB, D)
d_a = grad_out_cat[:, :SA, :]
d_b = grad_out_cat[:, SA:, :]

save("seq_cat2", "a.bin",            a.numpy())
save("seq_cat2", "b.bin",            b.numpy())
save("seq_cat2", "expected_out.bin", out_cat.numpy())
save("seq_cat2", "grad_out.bin",     grad_out_cat.numpy())
save("seq_cat2", "expected_da.bin",  d_a.numpy())
save("seq_cat2", "expected_db.bin",  d_b.numpy())

print(f"\n  B={B}  S={S}  N_HEADS={N_HEADS}  HD={HD}  D={D}  SA={SA}  SB={SB}")
print("done")
