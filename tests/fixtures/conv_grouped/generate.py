#!/usr/bin/env python3
"""Generate fixtures for the grouped Conv2d CUDA forward-pass test.

Two variants, both with BATCH=2, H=4, W=4, KH=KW=3, stride=1, pad=1:

  depthwise/   — groups = C_IN = C_OUT = 128  (depthwise conv, weight [128,1,3,3])
  g4/          — groups = 4, C_IN = 8, C_OUT = 8  (weight [8,2,3,3])

For each variant we save:
  x.bin                — input  (B, C_IN, H, W)  NCHW float32
  conv_w.bin           — weight (C_OUT, C_IN/G, KH, KW)  float32
  expected_output.bin  — output (B, C_OUT, H, W)  NCHW float32

All tensors are row-major float32 (little-endian).
"""

import os
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)

BASE  = os.path.dirname(os.path.abspath(__file__))
BATCH, H, W = 2, 4, 4


def save(path, tensor):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    arr.tofile(path)
    print(f"  {os.path.relpath(path, BASE):50s}  {list(arr.shape)}")


def gen(subdir, c_in, c_out, groups):
    out_dir = os.path.join(BASE, subdir)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(42)
    conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1,
                     groups=groups, bias=False)

    x = torch.randn(BATCH, c_in, H, W)
    with torch.no_grad():
        y = conv(x)

    print(f"\n[groups={groups} c_in={c_in} c_out={c_out}  weight {list(conv.weight.shape)}]")
    save(os.path.join(out_dir, "x.bin"),               x)
    save(os.path.join(out_dir, "conv_w.bin"),          conv.weight)
    save(os.path.join(out_dir, "expected_output.bin"), y)


gen("depthwise", c_in=128, c_out=128, groups=128)
gen("g4",        c_in=8,   c_out=8,   groups=4)

print("\ndone.")
