#!/usr/bin/env python3
"""Generate fixtures for the YOLO BCE classification loss CUDA forward test.

Layout (channels-first):
    pred:   [C, N]  — predicted class logits per anchor
    target: [C, N]  — soft class labels ∈ [0, 1] per anchor
    loss:   [N]     — per-anchor BCE loss (summed over C classes)

BCE (numerically stable):
    loss_cn = max(x, 0) − x·t + log(1 + exp(−|x|))
    loss_n  = Σ_c  loss_cn

Files written (all flat little-endian f32):
    pred.bin     — logits [C, N] flattened
    target.bin   — soft labels [C, N] flattened
    expected.bin — per-anchor BCE loss [N]

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


def bce_loss_sum(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-anchor BCE summed over classes.

    Args:
        pred_logits: [N, C]
        target:      [N, C]
    Returns:
        loss: [N]
    """
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")  # [N, C]
    return bce.sum(dim=1)  # [N]


# Pred logits: unit-normal random.
pred_NC = torch.randn(N, C)

# Soft targets: mostly zeros with a few positives, values in [0, 1].
target_NC = torch.zeros(N, C)
for n in range(N):
    # 1-3 positive classes per anchor
    k = torch.randint(1, 4, (1,)).item()
    pos_idx = torch.randperm(C)[:k]
    target_NC[n, pos_idx] = torch.rand(k) * 0.9 + 0.1  # ∈ [0.1, 1.0]

with torch.no_grad():
    loss = bce_loss_sum(pred_NC, target_NC)  # [N]

print(f"Saving fixtures to {BASE}")
print(f"  N={N}, C={C}")

# Kernel expects channels-first layout [C, N].
pred_CN   = pred_NC.T.contiguous()    # [C, N]
target_CN = target_NC.T.contiguous()  # [C, N]

save("pred.bin",     pred_CN.numpy())
save("target.bin",   target_CN.numpy())
save("expected.bin", loss.numpy())

print(f"\n  loss mean={loss.mean():.6f}  min={loss.min():.6f}  max={loss.max():.6f}")
print("done")
