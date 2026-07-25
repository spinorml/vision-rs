#!/usr/bin/env python3
"""Generate fixtures for sigmoid focal loss forward and backward CUDA tests.

Layout (flat 1-D):
  logits:  [N]  — predicted class logits
  targets: [N]  — binary targets in {0, 1} (as f32)
  loss:    [N]  — per-element focal loss

Files written (flat little-endian f32):
  logits.bin, targets.bin, expected_loss.bin,
  grad_loss.bin, expected_dlogits.bin

Usage:
  python tests/fixtures/sigmoid_focal_loss/generate.py
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))

N        = 128
ALPHA    = 0.25
GAMMA    = 2.0
NUM_BOXES = 8.0


def save(name: str, arr) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    path = os.path.join(BASE, name)
    arr.tofile(path)
    print(f"  {name:40s}  shape={list(arr.shape)}")


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                        alpha: float, gamma: float, num_boxes: float) -> torch.Tensor:
    """Per-element sigmoid focal loss matching the Triton kernel formula."""
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal_weight = (1.0 - p_t + 1e-8) ** gamma
    ce = -torch.log(p_t + 1e-8)
    return alpha_t * focal_weight * ce / num_boxes


# ── Generate inputs ───────────────────────────────────────────────────────────

logits  = torch.randn(N)
# Binary targets — 40% positive
targets = (torch.rand(N) < 0.4).float()

logits_t  = logits.requires_grad_(True)

loss = sigmoid_focal_loss(logits_t, targets, ALPHA, GAMMA, NUM_BOXES)

grad_loss = torch.randn_like(loss)
loss.backward(grad_loss)
d_logits = logits_t.grad.clone()

# ── Save fixtures ─────────────────────────────────────────────────────────────

print(f"\nSaving fixtures to {BASE}")
print(f"  N={N}  ALPHA={ALPHA}  GAMMA={GAMMA}  NUM_BOXES={NUM_BOXES}")
print()

save("logits.bin",          logits.detach().numpy())
save("targets.bin",         targets.numpy())
save("expected_loss.bin",   loss.detach().numpy())
save("grad_loss.bin",       grad_loss.numpy())
save("expected_dlogits.bin", d_logits.numpy())

print(f"\n  loss  mean={loss.detach().mean():.6f}")
print(f"  grad  max_abs={d_logits.abs().max():.6f}")
print("done")
