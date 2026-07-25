#!/usr/bin/env python3
"""Generate fixtures for ms_deform_attn forward and backward CUDA tests.

Implements the same bilinear sampling convention as the Triton kernel:
  px = lx * (Wl - 1),  py = ly * (Hl - 1)
  (align_corners=True-style pixel coordinates, matching the rf-detr reference)

Layouts (all row-major):
  value:          [BH, S_total, HEAD_DIM]
  sampling_locs:  [BH, Nq, n_levels, n_points, 2]   (lx, ly in [0, 1])
  attn_weights:   [BH, Nq, n_levels * n_points]      (post-softmax, sum=1 per query-head)
  spatial_shapes: [n_levels, 2]                       (Hl, Wl) stored as f32
  level_start:    [n_levels]                          cumulative start indices as f32
  output:         [BH, Nq, HEAD_DIM]

Files written (flat little-endian f32):
  value.bin, sampling_locs.bin, attn_weights.bin
  spatial_shapes.bin, level_start.bin
  expected_output.bin
  grad_output.bin, expected_dvalue.bin,
  expected_dsampling_locs.bin, expected_dattn_weights.bin

Usage:
  python tests/fixtures/ms_deform_attn/generate.py
"""

import os
import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Dimensions ────────────────────────────────────────────────────────────────

BH       = 4   # batch × heads  (e.g. B=2, H=2)
NQ       = 6   # number of queries
N_LEVELS = 2
N_POINTS = 4
HEAD_DIM = 8

# Level spatial shapes: level 0 = 4×4, level 1 = 2×2
SPATIAL_SHAPES = [(4, 4), (2, 2)]   # (H_l, W_l)
S_TOTAL = sum(h * w for h, w in SPATIAL_SHAPES)  # 16 + 4 = 20
LEVEL_STARTS = []
start = 0
for h, w in SPATIAL_SHAPES:
    LEVEL_STARTS.append(start)
    start += h * w


def save(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    path = os.path.join(BASE, name)
    arr.tofile(path)
    print(f"  {name:40s}  shape={list(arr.shape)}")


# ── Reference bilinear sampling ───────────────────────────────────────────────

def bilinear_sample_reference(value, sampling_locs, attn_weights, spatial_shapes, level_starts):
    """Pure PyTorch reference implementation matching the Triton kernel exactly.

    Args:
        value:          [BH, S_total, HEAD_DIM]
        sampling_locs:  [BH, NQ, n_levels, n_points, 2]  lx, ly in [0, 1]
        attn_weights:   [BH, NQ, n_levels * n_points]
        spatial_shapes: list of (H_l, W_l)
        level_starts:   list of start indices

    Returns:
        output: [BH, NQ, HEAD_DIM]
    """
    bh, nq, n_levels, n_points, _ = sampling_locs.shape
    output = torch.zeros(bh, nq, HEAD_DIM, dtype=torch.float32)

    for l, ((H_l, W_l), start_l) in enumerate(zip(spatial_shapes, level_starts)):
        for p in range(n_points):
            # [BH, NQ]
            lx = sampling_locs[:, :, l, p, 0]
            ly = sampling_locs[:, :, l, p, 1]

            # Pixel coordinates
            px = lx * (W_l - 1)
            py = ly * (H_l - 1)

            x0 = torch.floor(px).long()
            y0 = torch.floor(py).long()
            x1 = x0 + 1
            y1 = y0 + 1

            wx1 = (px - x0.float())      # [BH, NQ]
            wy1 = (py - y0.float())
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1

            # Boundary masks (zero out-of-bounds)
            mask_x0 = ((x0 >= 0) & (x0 < W_l)).float()
            mask_x1 = ((x1 >= 0) & (x1 < W_l)).float()
            mask_y0 = ((y0 >= 0) & (y0 < H_l)).float()
            mask_y1 = ((y1 >= 0) & (y1 < H_l)).float()

            wx0_m = wx0 * mask_x0
            wx1_m = wx1 * mask_x1
            wy0_m = wy0 * mask_y0
            wy1_m = wy1 * mask_y1

            # Clamp for safe gather
            x0_s = torch.clamp(x0, 0, W_l - 1)
            x1_s = torch.clamp(x1, 0, W_l - 1)
            y0_s = torch.clamp(y0, 0, H_l - 1)
            y1_s = torch.clamp(y1, 0, H_l - 1)

            # Spatial flat indices into value buffer
            idx00 = start_l + y0_s * W_l + x0_s   # [BH, NQ]
            idx01 = start_l + y0_s * W_l + x1_s
            idx10 = start_l + y1_s * W_l + x0_s
            idx11 = start_l + y1_s * W_l + x1_s

            # Gather: value[:, idx, :]  →  [BH, NQ, HEAD_DIM]
            bh_idx = torch.arange(bh)[:, None].expand(bh, nq)
            v00 = value[bh_idx, idx00, :]
            v01 = value[bh_idx, idx01, :]
            v10 = value[bh_idx, idx10, :]
            v11 = value[bh_idx, idx11, :]

            # Bilinear: weight is [BH, NQ, 1] broadcast
            bilinear = (
                (wy0_m * wx0_m)[..., None] * v00 +
                (wy0_m * wx1_m)[..., None] * v01 +
                (wy1_m * wx0_m)[..., None] * v10 +
                (wy1_m * wx1_m)[..., None] * v11
            )

            # Attention weight [BH, NQ]
            w = attn_weights[:, :, l * n_points + p]

            output += w[..., None] * bilinear

    return output


# ── Generate inputs ───────────────────────────────────────────────────────────

value_data = torch.randn(BH, S_TOTAL, HEAD_DIM)

# Sampling locations in [0.05, 0.95] to keep near centre and avoid extreme OOB
sampling_locs_data = torch.rand(BH, NQ, N_LEVELS, N_POINTS, 2) * 0.9 + 0.05

# Attention weights: softmax over n_levels * n_points per query-head
attn_logits = torch.randn(BH, NQ, N_LEVELS * N_POINTS)
attn_weights_data = torch.softmax(attn_logits, dim=-1)

spatial_shapes_arr = np.array(SPATIAL_SHAPES, dtype=np.float32)    # [n_levels, 2]
level_start_arr    = np.array(LEVEL_STARTS,  dtype=np.float32)     # [n_levels]

# ── Forward reference ─────────────────────────────────────────────────────────

value_t = value_data.requires_grad_(True)
slocs_t = sampling_locs_data.requires_grad_(True)
aw_t    = attn_weights_data.requires_grad_(True)

output = bilinear_sample_reference(
    value_t, slocs_t, aw_t, SPATIAL_SHAPES, LEVEL_STARTS
)

# ── Backward reference ────────────────────────────────────────────────────────

grad_output = torch.randn_like(output)
output.backward(grad_output)

d_value         = value_t.grad.clone()
d_sampling_locs = slocs_t.grad.clone()
d_attn_weights  = aw_t.grad.clone()

# ── Save fixtures ─────────────────────────────────────────────────────────────

print(f"\nSaving fixtures to {BASE}")
print(f"  BH={BH}  NQ={NQ}  N_LEVELS={N_LEVELS}  N_POINTS={N_POINTS}  HEAD_DIM={HEAD_DIM}")
print(f"  SPATIAL_SHAPES={SPATIAL_SHAPES}  S_TOTAL={S_TOTAL}")
print(f"  LEVEL_STARTS={LEVEL_STARTS}")
print()

save("value.bin",               value_data.detach().numpy())
save("sampling_locs.bin",       sampling_locs_data.detach().numpy())
save("attn_weights.bin",        attn_weights_data.detach().numpy())
save("spatial_shapes.bin",      spatial_shapes_arr)
save("level_start.bin",         level_start_arr)
save("expected_output.bin",     output.detach().numpy())
save("grad_output.bin",         grad_output.numpy())
save("expected_dvalue.bin",     d_value.numpy())
save("expected_dsampling_locs.bin", d_sampling_locs.numpy())
save("expected_dattn_weights.bin",  d_attn_weights.numpy())

print(f"\n  output  mean_abs={output.abs().mean():.6f}")
print(f"  d_value max_abs={d_value.abs().max():.6f}")
print("done")
