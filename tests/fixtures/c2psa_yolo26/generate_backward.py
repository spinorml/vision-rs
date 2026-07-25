#!/usr/bin/env python3
"""Generate backward-pass fixtures for the C2PSA CUDA backward test.

C2PSA(c1=256, c2=256, n=1, e=0.5) — training mode, BATCH=2, H=4, W=4.

Runs a forward pass, captures BN intermediates (pre-BN NC, batch mean/rstd,
pre-SiLU NC) for every Conv+BN stage, derives Flash-Attention-2 tensors in
[BH, N, HEAD_DIM=32] layout, then backpropagates to produce dx_expected.

Output: tests/fixtures/c2psa_yolo26/backward/

Architecture (training mode):
  cv1:   Conv(256→256, 1×1) + BN(256) + SiLU
  split: a=(B,128,H,W)  b=(B,128,H,W)
  PSABlock on b:
    qkv:   Conv(128→256, 1×1) + BN(256)              [no SiLU]
    attn:  FA2(Q, K, V_lo) concat FA2(Q, K, V_hi)   [HEAD_DIM=32 each]
    pe:    DW-Conv(128→128, 3×3, G=128) + BN(128)    [no SiLU]
    proj:  Conv(128→128, 1×1) + BN(128)              [no SiLU]
    b = b + proj_out
    ffn0:  Conv(128→256, 1×1) + BN(256) + SiLU
    ffn1:  Conv(256→128, 1×1) + BN(128)              [no SiLU]
    b = b + ffn1_out
  cat(a, b)
  cv2:   Conv(256→256, 1×1) + BN(256) + SiLU

Flash-Attention-2 layout  [BH=4, N=16, D] row-major:
  BH = BATCH * NUM_HEADS = 2 * 2 = 4    N = H * W = 16
  Channel layout inside qkv BN output per head h (HEAD_SPAN=128):
    Q:    channels [h*128 + 0  : h*128 + 32]   → q_fa2  [4,16,32]
    K:    channels [h*128 + 32 : h*128 + 64]   → k_fa2  [4,16,32]
    V_lo: channels [h*128 + 64 : h*128 + 96]   → v_lo   [4,16,32]
    V_hi: channels [h*128 + 96 : h*128 +128]   → v_hi   [4,16,32]
  O_lo, O_hi: FA2 forward outputs        [4,16,32] each
  L:          log-sum-exp from FA2 fwd   [4,16]  (same for both calls)
"""

import os
import numpy as np
import torch
from ultralytics.nn.modules.block import C2PSA

SEED = 42
BASE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "backward")

BATCH     = 2
H, W      = 4, 4
C1        = 256   # outer channels
C         = 128   # hidden channels  (C1 * e = 128)
NUM_HEADS = 2
KEY_DIM   = 32    # q/k dim per head
HEAD_DIM  = 64    # v dim per head
HEAD_SPAN = KEY_DIM + KEY_DIM + HEAD_DIM  # 128 per head in the qkv projection
BH        = BATCH * NUM_HEADS  # 4
N         = H * W               # 16
SCALE     = KEY_DIM ** -0.5     # 1/sqrt(32)  ≈ 0.17678
BN_EPS    = 1e-5


# ── helpers ───────────────────────────────────────────────────────────────────

def save(name, tensor):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    arr.tofile(path)
    print(f"  {name:60s}  {list(arr.shape)}")


def save_conv_block(prefix, blk):
    """Save Conv2d weight and all four BN parameters."""
    save(f"{prefix}_conv_w.bin", blk.conv.weight)
    save(f"{prefix}_bn_w.bin",   blk.bn.weight)
    save(f"{prefix}_bn_b.bin",   blk.bn.bias)
    save(f"{prefix}_bn_rm.bin",  blk.bn.running_mean)
    save(f"{prefix}_bn_rv.bin",  blk.bn.running_var)


def nchw_to_nc(x_nchw):
    """(B, C, H, W) → (B*H*W, C) row-major NC."""
    B, C, H, W = x_nchw.shape
    return x_nchw.permute(0, 2, 3, 1).contiguous().reshape(-1, C)


# ── model + hooks ─────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
model = C2PSA(C1, C1, n=1, e=0.5)
model.train()   # training mode — BN uses batch statistics

attn = model.m[0].attn

intermediates = {}

def make_conv_hook(name):
    """Captures Conv2d output (= BN input) in NCHW layout."""
    def hook(module, inp, out):
        intermediates[f"{name}_pre_bn_nchw"] = out.detach().clone()
    return hook


def make_bn_hook(name):
    """Captures batch mean/rstd from BN input and BN output (pre-act)."""
    def hook(module, inp, out):
        x_in = inp[0]  # (B, C, H, W), BN input
        mean = x_in.mean(dim=(0, 2, 3))
        var  = x_in.var(dim=(0, 2, 3), unbiased=False)
        rstd = (var + BN_EPS).rsqrt()
        intermediates[f"{name}_bn_mean"]       = mean.detach()
        intermediates[f"{name}_bn_rstd"]       = rstd.detach()
        intermediates[f"{name}_pre_silu_nchw"] = out.detach().clone()
    return hook


STAGES = [
    ("cv1",          model.cv1),
    ("m0_attn_qkv",  attn.qkv),
    ("m0_attn_pe",   attn.pe),
    ("m0_attn_proj", attn.proj),
    ("m0_ffn0",      model.m[0].ffn[0]),
    ("m0_ffn1",      model.m[0].ffn[1]),
    ("cv2",          model.cv2),
]
for tag, blk in STAGES:
    blk.conv.register_forward_hook(make_conv_hook(tag))
    blk.bn.register_forward_hook(make_bn_hook(tag))

# ── forward pass ──────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
x = torch.randn(BATCH, C1, H, W, requires_grad=True)
y = model(x)

# ── derive FA2 tensors from captured qkv BN output ───────────────────────────
#
# qkv BN output: (B, QKV_H=256, H, W)
# Reshape to (B, NUM_HEADS, HEAD_SPAN=128, N) then split Q/K/V.

qkv_nchw = intermediates["m0_attn_qkv_pre_silu_nchw"]          # (2, 256, 4, 4)
qkv_bhdn = qkv_nchw.view(BATCH, NUM_HEADS, HEAD_SPAN, N)       # (2, 2, 128, 16)

q_bhdn = qkv_bhdn[:, :, :KEY_DIM,          :]   # (2,2,32,16)
k_bhdn = qkv_bhdn[:, :, KEY_DIM:2*KEY_DIM, :]   # (2,2,32,16)
v_bhdn = qkv_bhdn[:, :, 2*KEY_DIM:,        :]   # (2,2,64,16)

# FA2 layout: (BH, N, D) = contiguous row-major.
q_fa2  = q_bhdn.permute(0, 1, 3, 2).contiguous().reshape(BH, N, KEY_DIM)  # (4,16,32)
k_fa2  = k_bhdn.permute(0, 1, 3, 2).contiguous().reshape(BH, N, KEY_DIM)  # (4,16,32)
v_lo   = v_bhdn[:, :, :32, :].permute(0, 1, 3, 2).contiguous().reshape(BH, N, 32)
v_hi   = v_bhdn[:, :, 32:, :].permute(0, 1, 3, 2).contiguous().reshape(BH, N, 32)

# Attention scores and log-sum-exp.
# scores[bh, qi, ki] = (Q[bh,qi,:] · K[bh,ki,:]) * scale
scores = (q_fa2 @ k_fa2.transpose(1, 2)) * SCALE   # (4,16,16)
L      = torch.logsumexp(scores, dim=-1)            # (4,16)  — FA2 LSE
attn_w = scores.softmax(dim=-1)                     # (4,16,16)

# Attention outputs in FA2 layout (split V along the 64-dim axis).
o_lo = attn_w @ v_lo  # (4,16,32)
o_hi = attn_w @ v_hi  # (4,16,32)

# ── backward pass ─────────────────────────────────────────────────────────────

torch.manual_seed(43)
dy = torch.randn_like(y)
y.backward(dy)
dx = x.grad.detach()

# ── save all fixtures ─────────────────────────────────────────────────────────

print(f"\nC2PSA backward  BATCH={BATCH}  H={H}  W={W}")
print(f"Output: {OUT_DIR}\n")

# Inputs / outputs
save("x.bin",               x.detach())
save("dy.bin",              dy)
save("dx_expected.bin",     dx)
save("expected_output.bin", y.detach())
print()

# Conv weights + BN parameters for all stages
for tag, blk in STAGES:
    save_conv_block(tag, blk)
print()

# BN forward intermediates (NC layout) for every stage.
# SiLU stages: cv1, m0_ffn0, cv2  — pre_silu_nc is needed for SiluBackward.
# Non-SiLU stages: qkv, pe, proj, ffn1 — pre_silu_nc is the BN output itself.
for tag, _ in STAGES:
    pre_bn_nc   = nchw_to_nc(intermediates[f"{tag}_pre_bn_nchw"])
    pre_silu_nc = nchw_to_nc(intermediates[f"{tag}_pre_silu_nchw"])
    save(f"{tag}_pre_bn_nc.bin",   pre_bn_nc)
    save(f"{tag}_bn_mean.bin",     intermediates[f"{tag}_bn_mean"])
    save(f"{tag}_bn_rstd.bin",     intermediates[f"{tag}_bn_rstd"])
    save(f"{tag}_pre_silu_nc.bin", pre_silu_nc)
print()

# FA2 tensors — layout [BH=4, N=16, D=32] or [BH=4, N=16] for L.
save("attn_q.bin",    q_fa2)
save("attn_k.bin",    k_fa2)
save("attn_v_lo.bin", v_lo)
save("attn_v_hi.bin", v_hi)
save("attn_o_lo.bin", o_lo)
save("attn_o_hi.bin", o_hi)
save("attn_l.bin",    L)
print()

print(f"done — {len(os.listdir(OUT_DIR))} files written")
