"""Generate Conv block training fixtures for the YOLO26n (nano) stem.

Uses ultralytics Conv in training mode: Conv2d → BatchNorm2d (batch stats) → SiLU.
Parameters match the nano variant stem: c1=3, c2=16, k=3, s=2, autopad=1.

The BN Triton kernel expects (N, C) channels-last layout, where N = B×H×W.
The test therefore transposes the conv output from (B,C,H,W) → (N,C) on the
host before the BN launch, and transposes expected_output to NC for comparison.

Files written (all flat little-endian f32):
  x.bin               — input  (2, 3, 16, 16) NCHW
  conv_weight.bin     — Conv2d weight  (16, 3, 3, 3)
  bn_weight.bin       — BN γ           (16,)
  bn_bias.bin         — BN β           (16,)
  bn_running_mean.bin — initial running μ  before forward  (16,)
  bn_running_var.bin  — initial running σ² before forward  (16,)
  expected_output.bin — forward output (2, 16, 8, 8) NCHW, training-mode BN

Run from the repo root:
    python tests/fixtures/conv_yolo26/generate.py
"""

import os
import torch
from ultralytics.nn.modules.conv import Conv

HERE = os.path.dirname(os.path.abspath(__file__))

BATCH = 2
C_IN  = 3
C_OUT = 16
H, W  = 16, 16
K, S  = 3, 2

torch.manual_seed(42)

model = Conv(C_IN, C_OUT, k=K, s=S)
model.train()  # BN uses per-batch statistics, matches BatchNormStatsForward kernel

# Save initial running stats BEFORE the forward pass so the test can pass the
# same initial values to batch_norm_stats_forward (which updates them via EMA).
initial_rm = model.bn.running_mean.clone()
initial_rv = model.bn.running_var.clone()

x = torch.randn(BATCH, C_IN, H, W)

with torch.no_grad():
    y = model(x)

def save(name: str, t: torch.Tensor) -> None:
    path = os.path.join(HERE, name)
    t.detach().contiguous().cpu().numpy().astype("float32").tofile(path)
    print(f"  {name}: shape={tuple(t.shape)}, numel={t.numel()}")

print("Generating conv_yolo26 fixtures (training mode):")
save("x.bin",               x)
save("conv_weight.bin",     model.conv.weight)
save("bn_weight.bin",       model.bn.weight)
save("bn_bias.bin",         model.bn.bias)
save("bn_running_mean.bin", initial_rm)
save("bn_running_var.bin",  initial_rv)
save("expected_output.bin", y)

out_h = (H + 2 * 1 - K) // S + 1  # 8
out_w = (W + 2 * 1 - K) // S + 1  # 8
n_bn  = BATCH * out_h * out_w      # 128 — BN "rows" (B×H_out×W_out per channel)
print(f"\nInput:  ({BATCH}, {C_IN}, {H}, {W})  = {BATCH * C_IN * H * W} elements")
print(f"Output: ({BATCH}, {C_OUT}, {out_h}, {out_w}) = {BATCH * C_OUT * out_h * out_w} elements")
print(f"N_BN (samples per BN channel): {BATCH}×{out_h}×{out_w} = {n_bn}")
print(f"BN eps={model.bn.eps}, momentum={model.bn.momentum}")
