#!/usr/bin/env python3
"""Generate fixtures for the YOLO CIoU loss CUDA forward test.

Layout (channels-first):
    pred:   [4, N]  — predicted (cx, cy, w, h) per anchor
    target: [4, N]  — target (cx, cy, w, h) per anchor
    loss:   [N]     — per-anchor CIoU loss

CIoU = 1 − IoU + d²/c² + α·v
where v = (4/π²) · (atan(tw/th) − atan(pw/ph))²
      α = v / (1 − IoU + v + ε)

Files written (all flat little-endian f32):
    pred.bin     — predicted boxes [4, N] flattened
    target.bin   — target boxes [4, N] flattened
    expected.bin — per-anchor CIoU loss [N]

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

def save(name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    path = os.path.join(BASE, name)
    arr.tofile(path)
    print(f"  {name:30s}  shape={list(arr.shape)}  dtype={arr.dtype}")


def ciou_loss(pred_xywh: torch.Tensor, target_xywh: torch.Tensor, eps: float = 1e-7):
    """Compute per-anchor CIoU loss.

    Args:
        pred_xywh:   [N, 4] (cx, cy, w, h)
        target_xywh: [N, 4] (cx, cy, w, h)
    Returns:
        loss: [N]
    """
    # Unpack
    px, py, pw, ph = pred_xywh[:, 0], pred_xywh[:, 1], pred_xywh[:, 2], pred_xywh[:, 3]
    tx, ty, tw, th = target_xywh[:, 0], target_xywh[:, 1], target_xywh[:, 2], target_xywh[:, 3]

    # Corners
    px1, px2 = px - pw / 2, px + pw / 2
    py1, py2 = py - ph / 2, py + ph / 2
    tx1, tx2 = tx - tw / 2, tx + tw / 2
    ty1, ty2 = ty - th / 2, ty + th / 2

    # Intersection
    ix1 = torch.maximum(px1, tx1)
    ix2 = torch.minimum(px2, tx2)
    iy1 = torch.maximum(py1, ty1)
    iy2 = torch.minimum(py2, ty2)
    inter_w = torch.clamp(ix2 - ix1, min=0)
    inter_h = torch.clamp(iy2 - iy1, min=0)
    inter = inter_w * inter_h

    # Union
    union = pw * ph + tw * th - inter

    # IoU
    iou = inter / (union + eps)

    # Center distance squared
    d2 = (px - tx) ** 2 + (py - ty) ** 2

    # Smallest enclosing box diagonal squared
    ex1, ex2 = torch.minimum(px1, tx1), torch.maximum(px2, tx2)
    ey1, ey2 = torch.minimum(py1, ty1), torch.maximum(py2, ty2)
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2

    # Aspect-ratio term
    v = (4 / math.pi ** 2) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))) ** 2
    alpha = v / (1 - iou + v + eps)

    # CIoU loss
    loss = 1 - iou + d2 / (c2 + eps) + alpha * v
    return loss


# Generate random boxes in a plausible image domain (640×640).
# Widths and heights must be positive.
cx  = torch.rand(N) * 600 + 20      # center x ∈ [20, 620]
cy  = torch.rand(N) * 600 + 20      # center y ∈ [20, 620]
pw  = torch.rand(N) * 200 + 10      # pred width  ∈ [10, 210]
ph  = torch.rand(N) * 200 + 10      # pred height ∈ [10, 210]
pred_xywh = torch.stack([cx, cy, pw, ph], dim=1)  # [N, 4]

# Targets: perturb pred a bit so they're close but distinct.
tx = cx + torch.randn(N) * 20
ty = cy + torch.randn(N) * 20
tw = torch.clamp(pw + torch.randn(N) * 30, min=1.0)
th = torch.clamp(ph + torch.randn(N) * 30, min=1.0)
target_xywh = torch.stack([tx, ty, tw, th], dim=1)  # [N, 4]

with torch.no_grad():
    loss = ciou_loss(pred_xywh, target_xywh)

print(f"Saving fixtures to {BASE}")
print(f"  N={N}")

# Kernel expects channels-first layout [4, N].
pred_cf   = pred_xywh.T.contiguous()    # [4, N]
target_cf = target_xywh.T.contiguous()  # [4, N]

save("pred.bin",     pred_cf.numpy())
save("target.bin",   target_cf.numpy())
save("expected.bin", loss.numpy())

print(f"\n  loss mean={loss.mean():.6f}  min={loss.min():.6f}  max={loss.max():.6f}")
print("done")
