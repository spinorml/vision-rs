#!/usr/bin/env python3
"""Generate fixtures for the YOLO BCE classification loss CUDA forward and backward tests.

Layout (channels-first):
    pred:   [C, N]  — predicted class logits per anchor
    target: [C, N]  — soft class labels ∈ [0, 1] per anchor
    loss:   [N]     — per-anchor BCE loss (summed over C classes)

Backward fixtures:
    dy:             [N]     — random upstream gradient
    expected_dpred: [C, N]  — ∂L/∂pred from PyTorch autograd

Files written (all flat little-endian f32):
    pred.bin            target.bin          expected.bin
    dy.bin              expected_dpred.bin

Usage:
    python tests/fixtures/yolo_bce_cls_loss/generate.py
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
N = 32   # number of anchors
C = 80   # number of classes (COCO default)


def save(name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    path = os.path.join(BASE, name)
    arr.tofile(path)
    print(f"  {name:30s}  shape={list(arr.shape)}  dtype={arr.dtype}")


# Pred logits with grad tracking.
pred_NC = torch.randn(N, C).requires_grad_(True)  # [N, C]

# Soft targets: mostly zeros with a few positives per anchor.
target_NC = torch.zeros(N, C)
for n in range(N):
    k = torch.randint(1, 4, (1,)).item()
    pos_idx = torch.randperm(C)[:k]
    target_NC[n, pos_idx] = torch.rand(k) * 0.9 + 0.1  # ∈ [0.1, 1.0]

# Forward: per-anchor BCE summed over classes.
bce_per_element = F.binary_cross_entropy_with_logits(pred_NC, target_NC, reduction="none")  # [N, C]
loss = bce_per_element.sum(dim=1)  # [N]

# Upstream gradient — random [N].
torch.manual_seed(7)
dy = torch.rand(N) * 2 - 1  # uniform in [−1, 1]

# Backward pass.
loss.backward(dy)
expected_dpred = pred_NC.grad  # [N, C]

print(f"Saving fixtures to {BASE}")
print(f"  N={N}, C={C}")

# Channels-first layout [C, N] for pred, target, d_pred.
pred_CN   = pred_NC.detach().T.contiguous()   # [C, N]
target_CN = target_NC.T.contiguous()          # [C, N]
dpred_CN  = expected_dpred.T.contiguous()     # [C, N]

save("pred.bin",           pred_CN.numpy())
save("target.bin",         target_CN.numpy())
save("expected.bin",       loss.detach().numpy())
save("dy.bin",             dy.numpy())
save("expected_dpred.bin", dpred_CN.numpy())

print(f"\n  loss mean={loss.detach().mean():.6f}  min={loss.detach().min():.6f}  max={loss.detach().max():.6f}")
print(f"  grad max_abs={expected_dpred.abs().max():.6f}")
print("done")
