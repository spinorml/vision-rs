"""Generate Softmax forward/backward fixtures for a YOLO detection head.

Represents the P5 (20×20) detection head of YOLOv8 at 640×640 input:
  - batch = 2
  - anchors = 20 × 20 = 400 per image
  - n_rows = batch × anchors = 800   (one softmax row per anchor)
  - n_cols = 128                      (80 COCO classes padded to the next
                                       power of 2 — required by the kernel)

Softmax is applied over the class-score dimension (dim=-1 / dim=1 in 2-D view).
Stored as flat row-major little-endian f32.

Run from the repo root:
    python tests/fixtures/softmax_yolo/generate.py
"""

import os
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

BATCH    = 2
ANCHORS  = 20 * 20   # P5 feature map
N_ROWS   = BATCH * ANCHORS   # 800
N_COLS   = 128               # 80 COCO classes → next power of 2

torch.manual_seed(42)
# Logits in a realistic range for detection class scores.
x  = torch.empty(N_ROWS, N_COLS).uniform_(-3.0, 3.0)
dy = torch.empty(N_ROWS, N_COLS).uniform_(-1.0, 1.0)

x_r = x.clone().requires_grad_(True)
y   = F.softmax(x_r, dim=-1)
y.backward(dy)
dx  = x_r.grad.detach()

def save(path: str, t: torch.Tensor) -> None:
    t.detach().contiguous().cpu().numpy().astype("float32").tofile(path)

save(f"{HERE}/x.bin",                x)
save(f"{HERE}/dy.bin",               dy)
save(f"{HERE}/expected_forward.bin", y.detach())
save(f"{HERE}/expected_backward.bin", dx)

print(f"Generated fixtures: n_rows={N_ROWS}, n_cols={N_COLS}, total={N_ROWS * N_COLS} elements")
