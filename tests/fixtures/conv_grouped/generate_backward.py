#!/usr/bin/env python3
"""Generate backward-pass fixtures for the grouped Conv2d CUDA tests.

Loads forward fixtures (x.bin, conv_w.bin) produced by generate.py, runs a
forward pass with requires_grad=True, then backpropagates through a random
upstream gradient to obtain reference dx and dw from PyTorch autograd.

Two variants — same shapes as generate.py:
  depthwise/   groups = C_IN = C_OUT = 128, weight [128, 1, 3, 3]
  g4/          groups = 4, C_IN = 8, C_OUT = 8, weight [8, 2, 3, 3]

For each variant we save:
  dy.bin           — upstream gradient  (B, C_OUT, H, W)       float32 NCHW
  dx_expected.bin  — reference dx       (B, C_IN,  H, W)       float32 NCHW
  dw_expected.bin  — reference dw       (C_OUT, C_IN/G, 3, 3)  float32
"""

import os
import numpy as np
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.abspath(__file__))
BATCH, H, W = 2, 4, 4


def save(path, tensor):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    arr.tofile(path)
    print(f"  {os.path.relpath(path, BASE):50s}  {list(arr.shape)}")


def gen_backward(subdir, c_in, c_out, groups):
    out_dir = os.path.join(BASE, subdir)

    # Reload the forward fixtures so x and w are identical to the forward test.
    x_flat = np.fromfile(os.path.join(out_dir, "x.bin"),      dtype=np.float32)
    w_flat = np.fromfile(os.path.join(out_dir, "conv_w.bin"), dtype=np.float32)

    x = torch.from_numpy(x_flat.reshape(BATCH, c_in, H, W)).requires_grad_(True)
    w = nn.Parameter(
        torch.from_numpy(w_flat.reshape(c_out, c_in // groups, 3, 3))
    )

    conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1,
                     groups=groups, bias=False)
    conv.weight = w
    y = conv(x)

    # Reproducible upstream gradient, seeded independently of generate.py.
    torch.manual_seed(43)
    dy = torch.randn_like(y)

    y.backward(dy)

    print(f"\n[backward groups={groups} c_in={c_in} c_out={c_out}  "
          f"weight {list(conv.weight.shape)}]")
    save(os.path.join(out_dir, "dy.bin"),           dy)
    save(os.path.join(out_dir, "dx_expected.bin"),  x.grad)
    save(os.path.join(out_dir, "dw_expected.bin"),  w.grad)


gen_backward("depthwise", c_in=128, c_out=128, groups=128)
gen_backward("g4",        c_in=8,   c_out=8,   groups=4)

print("\ndone.")
