#!/usr/bin/env python3
"""Generate fixtures for the YOLO CIoU loss CUDA forward and backward tests.

Layout (channels-first):
    pred:   [4, N]  — predicted (cx, cy, w, h) per anchor
    target: [4, N]  — target (cx, cy, w, h) per anchor
    loss:   [N]     — per-anchor CIoU loss

Saved activations (written by the forward kernel):
    iou:    [N]     — IoU per anchor
    v:      [N]     — aspect-ratio consistency term
    alpha:  [N]     — α coefficient

Backward fixtures:
    dy:             [N]     — random upstream gradient
    expected_dpred: [4, N]  — ∂L/∂pred from PyTorch autograd

Files written (all flat little-endian f32):
    pred.bin            target.bin          expected.bin
    iou.bin             v.bin               alpha.bin
    dy.bin              expected_dpred.bin

Usage:
    python tests/fixtures/yolo_ciou_loss/generate.py
"""

import os
import math
import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
N = 32  # number of anchors
EPS = 1e-7


def save(name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    path = os.path.join(BASE, name)
    arr.tofile(path)
    print(f"  {name:30s}  shape={list(arr.shape)}  dtype={arr.dtype}")


def ciou_loss_with_saved(pred_xywh: torch.Tensor, target_xywh: torch.Tensor, eps: float = EPS):
    """Compute per-anchor CIoU loss and return saved activations.

    Args:
        pred_xywh:   [N, 4] (cx, cy, w, h) — requires_grad
        target_xywh: [N, 4] (cx, cy, w, h) — no grad
    Returns:
        loss:  [N]
        iou:   [N]
        v:     [N]
        alpha: [N]
    """
    px, py, pw, ph = pred_xywh[:, 0], pred_xywh[:, 1], pred_xywh[:, 2], pred_xywh[:, 3]
    tx, ty, tw, th = target_xywh[:, 0], target_xywh[:, 1], target_xywh[:, 2], target_xywh[:, 3]

    px1, px2 = px - pw / 2, px + pw / 2
    py1, py2 = py - ph / 2, py + ph / 2
    tx1, tx2 = tx - tw / 2, tx + tw / 2
    ty1, ty2 = ty - th / 2, ty + th / 2

    ix1 = torch.maximum(px1, tx1)
    ix2 = torch.minimum(px2, tx2)
    iy1 = torch.maximum(py1, ty1)
    iy2 = torch.minimum(py2, ty2)
    inter_w = torch.clamp(ix2 - ix1, min=0)
    inter_h = torch.clamp(iy2 - iy1, min=0)
    inter = inter_w * inter_h

    union = pw * ph + tw * th - inter
    iou = inter / (union + eps)

    d2 = (px - tx) ** 2 + (py - ty) ** 2

    ex1, ex2 = torch.minimum(px1, tx1), torch.maximum(px2, tx2)
    ey1, ey2 = torch.minimum(py1, ty1), torch.maximum(py2, ty2)
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2

    v = (4 / math.pi ** 2) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))) ** 2
    alpha = v / (1 - iou + v + eps)

    loss = 1 - iou + d2 / (c2 + eps) + alpha * v
    return loss, iou.detach(), v.detach(), alpha.detach()


# ── Generate random boxes ─────────────────────────────────────────────────────

cx  = torch.rand(N) * 600 + 20
cy  = torch.rand(N) * 600 + 20
pw  = torch.rand(N) * 200 + 10
ph  = torch.rand(N) * 200 + 10
pred_xywh = torch.stack([cx, cy, pw, ph], dim=1).requires_grad_(True)  # [N, 4]

tx = cx.detach() + torch.randn(N) * 20
ty = cy.detach() + torch.randn(N) * 20
tw = torch.clamp(pw.detach() + torch.randn(N) * 30, min=1.0)
th = torch.clamp(ph.detach() + torch.randn(N) * 30, min=1.0)
target_xywh = torch.stack([tx, ty, tw, th], dim=1)  # [N, 4]

# Forward pass (retains graph for backward).
loss, iou, v, alpha = ciou_loss_with_saved(pred_xywh, target_xywh)

# Upstream gradient — random [N].
torch.manual_seed(7)
dy = torch.rand(N) * 2 - 1  # uniform in [−1, 1]

# Backward pass.
loss.backward(dy)
expected_dpred = pred_xywh.grad  # [N, 4]

print(f"Saving fixtures to {BASE}")
print(f"  N={N}")

# Channels-first layout [4, N] for pred, target, d_pred.
pred_cf   = pred_xywh.detach().T.contiguous()   # [4, N]
target_cf = target_xywh.T.contiguous()          # [4, N]
dpred_cf  = expected_dpred.T.contiguous()        # [4, N]

save("pred.bin",            pred_cf.numpy())
save("target.bin",          target_cf.numpy())
save("expected.bin",        loss.detach().numpy())
save("iou.bin",             iou.numpy())
save("v.bin",               v.numpy())
save("alpha.bin",           alpha.numpy())
save("dy.bin",              dy.numpy())
save("expected_dpred.bin",  dpred_cf.numpy())

print(f"\n  loss  mean={loss.detach().mean():.6f}")
print(f"  iou   mean={iou.mean():.6f}")
print(f"  grad  max_abs={expected_dpred.abs().max():.6f}")
print("done")
