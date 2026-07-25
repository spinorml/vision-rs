"""Generate Sigmoid forward/backward fixtures for a 4D (N, C, H, W) tensor.

Shape: (2, 8, 16, 16) = 4096 elements, stored as flat little-endian f32.

Run from the repo root:
    python tests/fixtures/sigmoid_4d/generate.py
"""

import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

N, C, H, W = 2, 8, 16, 16

torch.manual_seed(42)
x  = torch.empty(N, C, H, W).uniform_(-5.0, 5.0)
dy = torch.empty(N, C, H, W).uniform_(-2.0, 2.0)

x_r = x.clone().requires_grad_(True)
y   = torch.sigmoid(x_r)
y.backward(dy)
dx  = x_r.grad.detach()

def save(path: str, t: torch.Tensor) -> None:
    t.detach().contiguous().cpu().numpy().astype("float32").tofile(path)

save(f"{HERE}/x.bin",                x)
save(f"{HERE}/dy.bin",               dy)
save(f"{HERE}/expected_forward.bin", y.detach())
save(f"{HERE}/expected_backward.bin", dx)

print(f"Generated fixtures for shape ({N}, {C}, {H}, {W}) = {N*C*H*W} elements")
