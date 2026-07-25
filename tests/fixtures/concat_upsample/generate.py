#!/usr/bin/env python3
"""Generate fixtures for concat and upsample CUDA integration tests.

Concat layout: NCHW float32.
  B=2, C0=16, C1=16, C_total=32, H=4, W=4

Upsample layout: NCHW float32.
  B=2, C=16, H=4, W=4, SCALE=2, OH=8, OW=8
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(42)
BASE = os.path.dirname(os.path.abspath(__file__))

def save(name, arr):
    arr = np.ascontiguousarray(arr.detach().cpu().numpy().astype(np.float32))
    arr.tofile(os.path.join(BASE, name))
    print(f"  {name:45s}  {list(arr.shape)}")

B, C0, C1, H, W = 2, 16, 16, 4, 4
C_total = C0 + C1

# ── Concat forward ────────────────────────────────────────────────────────────
print("concat forward")
x0 = torch.empty(B, C0, H, W).uniform_(-5, 5).requires_grad_(True)
x1 = torch.empty(B, C1, H, W).uniform_(-5, 5).requires_grad_(True)
y_cat = torch.cat([x0, x1], dim=1)
save("x0.bin",                x0.detach())
save("x1.bin",                x1.detach())
save("expected_cat.bin",      y_cat.detach())

# ── Concat backward ───────────────────────────────────────────────────────────
print("concat backward")
torch.manual_seed(43)
dy = torch.empty(B, C_total, H, W).uniform_(-5, 5)
y_cat.backward(dy)
save("dy.bin",                dy)
save("expected_dx0.bin",      x0.grad)
save("expected_dx1.bin",      x1.grad)

# ── Upsample forward ──────────────────────────────────────────────────────────
print("upsample forward (reuses x0 as input)")
torch.manual_seed(44)
xu = torch.empty(B, C0, H, W).uniform_(-5, 5).requires_grad_(True)
yu = F.interpolate(xu, scale_factor=2, mode="nearest")
save("xu.bin",                xu.detach())
save("expected_up.bin",       yu.detach())

# ── Upsample backward ─────────────────────────────────────────────────────────
print("upsample backward")
torch.manual_seed(45)
dyu = torch.empty(B, C0, H * 2, W * 2).uniform_(-5, 5)
yu.backward(dyu)
save("dyu.bin",               dyu)
save("expected_dxu.bin",      xu.grad)
