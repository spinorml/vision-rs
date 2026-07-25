#!/usr/bin/env python3
"""
debug_train_grads.py  –  single-step training gradient dump for YOLO26n

Runs one forward + backward pass using ultralytics' E2ELoss on the first
COCO128 training image, then prints per-parameter gradient statistics.

Purpose: reference gradients to compare against vision-rs kernel outputs
and identify any kernel bugs causing gradient mismatches.

Usage:
    python scripts/debug_train_grads.py [--param PARAM_SUBSTR]

    --param   only print params whose name contains this substring
              (e.g. "model.0" for the first conv, "cv2.0" for a head branch)

Environment:
    DATASETS_CACHE_DIR  path to cached datasets (default: /mnt/data1/datasets/cache)
    YOLO26N_PT          path to yolo26n.pt  (default: ultralytics/yolo26n.pt in repo)
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_CACHE_DIR = Path(os.environ.get("DATASETS_CACHE_DIR", "/mnt/data1/datasets/cache"))
DEFAULT_PT = REPO_ROOT / "scripts" / "ultralytics" / "misc" / "yolo26n.pt"
IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def letterbox(img_bgr: np.ndarray, target: int = 640) -> np.ndarray:
    """Letterbox-resize to target×target, matching ultralytics preprocessing."""
    h, w = img_bgr.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top  = (target - new_h) // 2
    pad_left = (target - new_w) // 2
    out = np.full((target, target, 3), 114, dtype=np.uint8)
    out[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return out


def load_image(img_path: Path) -> torch.Tensor:
    """Load image → [1, 3, IMG_SIZE, IMG_SIZE] float32 in [0, 1] on DEVICE."""
    img_bgr = cv2.imread(str(img_path))
    img_lb  = letterbox(img_bgr, IMG_SIZE)
    img_rgb = img_lb[:, :, ::-1].copy()                   # BGR→RGB
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(DEVICE)                       # [1, 3, H, W]


def load_labels(label_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse YOLO-format label file → (batch_idx, cls, bboxes) tensors.

    Label file rows: class_id  cx  cy  w  h  (all normalised to [0, 1]).
    """
    classes, bboxes = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            classes.append(int(parts[0]))
            bboxes.append([float(p) for p in parts[1:5]])  # cx cy w h normalised

    n = len(classes)
    batch_idx = torch.zeros(n, dtype=torch.long,    device=DEVICE)
    cls       = torch.tensor(classes, dtype=torch.float32, device=DEVICE).view(-1, 1)
    bboxes_t  = torch.tensor(bboxes,   dtype=torch.float32, device=DEVICE)  # [N, 4]
    return batch_idx, cls, bboxes_t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--param",  default=None,
                        help="print only params containing this substring")
    parser.add_argument("--model",  default=str(DEFAULT_PT),
                        help="path to yolo26n.pt")
    parser.add_argument("--image",  default=None,
                        help="override first image (full path to .jpg)")
    args = parser.parse_args()

    # ── 1. Locate first training image ───────────────────────────────────────

    coco_train = DATASETS_CACHE_DIR / "coco128" / "train"
    if args.image:
        img_path = Path(args.image)
    else:
        images = sorted((coco_train / "images").glob("*.jpg"))
        assert images, f"No images found under {coco_train / 'images'}"
        img_path = images[0]

    label_path = coco_train / "labels" / (img_path.stem + ".txt")
    print(f"Image : {img_path.name}")
    print(f"Labels: {label_path.name}")
    orig = cv2.imread(str(img_path))
    print(f"Orig size : {orig.shape[1]}×{orig.shape[0]}")

    # ── 2. Load model ─────────────────────────────────────────────────────────

    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG
    print(f"\nLoading model from {args.model} ...")
    # Load without triggering fuse (fuse removes grad_fn).
    # ultralytics YOLO.predict fuses by default; load the raw torch model instead.
    import torch as _torch
    ckpt = _torch.load(args.model, map_location=DEVICE, weights_only=False)
    # The checkpoint stores the model under 'model' or is the model directly.
    raw = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = raw.float().to(DEVICE).train()
    model.requires_grad_(True)
    model.args = DEFAULT_CFG          # needed by init_criterion for hyp
    criterion = model.init_criterion()
    print(f"Model: {type(model).__name__}  device={DEVICE}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── 3. Build batch ────────────────────────────────────────────────────────

    img_t = load_image(img_path)
    batch_idx, cls, bboxes = load_labels(label_path)
    batch = {
        "img":       img_t,       # [1, 3, 640, 640]
        "batch_idx": batch_idx,   # [N] – all 0 (single image)
        "cls":       cls,         # [N, 1]
        "bboxes":    bboxes,      # [N, 4] normalised cx cy w h
    }
    print(f"\nBatch: 1 image, {len(batch_idx)} GT boxes")
    if len(batch_idx):
        print(f"  classes : {cls.view(-1).long().tolist()}")
        print(f"  bboxes  : {bboxes.tolist()}")

    # ── 4. Forward + loss ─────────────────────────────────────────────────────

    model.zero_grad()
    preds = model(img_t)

    loss_vec, loss_items = criterion(preds, batch)
    # loss_vec is shape [3] = [box_loss, cls_loss, dfl_loss] × batch_size × hyp gains
    scalar_loss = loss_vec.sum()
    print(f"\nLoss breakdown (batch-scaled, with hyp gains):")
    print(f"  box  = {loss_vec[0].item():.6f}  (hyp.box={DEFAULT_CFG.box})")
    print(f"  cls  = {loss_vec[1].item():.6f}  (hyp.cls={DEFAULT_CFG.cls})")
    print(f"  dfl  = {loss_vec[2].item():.6f}  (hyp.dfl={DEFAULT_CFG.dfl})")
    print(f"  total = {scalar_loss.item():.6f}")

    # ── 5. Backward ───────────────────────────────────────────────────────────

    scalar_loss.backward()

    # ── 6. Per-parameter gradient statistics ──────────────────────────────────

    print(f"\n{'─'*110}")
    hdr = f"{'Parameter':<55} {'Shape':<22} {'Norm':>12} {'Mean':>12} {'AbsMax':>12} {'HasGrad':>8}"
    print(hdr)
    print(f"{'─'*110}")

    filter_str = args.param
    rows_printed = 0

    for name, param in model.named_parameters():
        if filter_str and filter_str not in name:
            continue
        if param.grad is not None:
            g = param.grad.detach().float()
            norm   = g.norm().item()
            mean   = g.mean().item()
            absmax = g.abs().max().item()
            shape  = str(list(param.shape))
            print(f"{name:<55} {shape:<22} {norm:>12.6f} {mean:>12.6f} {absmax:>12.6f} {'yes':>8}")
        else:
            shape = str(list(param.shape))
            print(f"{name:<55} {shape:<22} {'—':>12} {'—':>12} {'—':>12} {'NO':>8}")
        rows_printed += 1

    print(f"{'─'*110}")
    print(f"Printed {rows_printed} parameters.")

    # ── 7. Summary ────────────────────────────────────────────────────────────

    all_params = list(model.named_parameters())
    n_with_grad  = sum(1 for _, p in all_params if p.grad is not None)
    n_zero_grad  = sum(
        1 for _, p in all_params
        if p.grad is not None and p.grad.abs().max().item() == 0.0
    )
    total_grad_norm = torch.cat([
        p.grad.detach().float().view(-1)
        for _, p in all_params if p.grad is not None
    ]).norm().item()

    print(f"\nSummary:")
    print(f"  params with grad   : {n_with_grad}/{len(all_params)}")
    print(f"  params with zero grad: {n_zero_grad}")
    print(f"  global gradient norm : {total_grad_norm:.6f}")


if __name__ == "__main__":
    main()
