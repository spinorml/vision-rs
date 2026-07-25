#!/usr/bin/env python3
"""
debug_infer.py  –  layer-by-layer inference output dump for YOLO26n

Loads YOLO26n in eval mode (with BN fusion to match Rust inference path),
registers forward hooks on every submodule, runs a single forward pass on a
val image, and prints per-layer output statistics.

Compare against:
    cargo run --example yolo26 --features cuda -- debug-infer \
        --model ultralytics/yolo26n --dataset assets/datasets/coco128.toml

Usage:
    python scripts/debug_infer.py [--model PATH] [--image-idx N] [--no-fuse]

Environment:
    DATASETS_CACHE_DIR  path to cached datasets
    YOLO26N_PT          overrides --model default path
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


# ---------------------------------------------------------------------------
# Image preprocessing (must match preprocess_image_raw in yolo26.rs)
# ---------------------------------------------------------------------------

def letterbox(img_bgr: np.ndarray, target: int = 640) -> np.ndarray:
    """Letterbox-resize to target×target, matching vision-rs preprocessing.

    Uses bilinear (INTER_LINEAR) resize and 114-gray padding — same as
    preprocess_image_raw in yolo26.rs (which uses Triangle/bilinear via the
    `image` crate).
    """
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
    """Load + preprocess to [1, 3, IMG_SIZE, IMG_SIZE] float32 in [0, 1]."""
    img_bgr = cv2.imread(str(img_path))
    assert img_bgr is not None, f"Could not read {img_path}"
    img_lb  = letterbox(img_bgr, IMG_SIZE)
    img_rgb = img_lb[:, :, ::-1].copy()                    # BGR → RGB
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(DEVICE)                        # [1, 3, H, W]


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def tensor_stats(t: torch.Tensor) -> dict:
    f = t.detach().float().flatten()
    return {
        "min":   f.min().item()      if f.numel() else float("nan"),
        "max":   f.max().item()      if f.numel() else float("nan"),
        "mean":  f.mean().item()     if f.numel() else float("nan"),
        "nans":  int(f.isnan().sum().item()),
        "infs":  int(f.isinf().sum().item()),
        "shape": list(t.shape),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",     default=str(DEFAULT_PT),
                        help="path to yolo26n.pt")
    parser.add_argument("--image-idx", type=int, default=0,
                        help="index of val image to use (default: 0)")
    parser.add_argument("--no-fuse",   action="store_true",
                        help="skip model.fuse() (for debugging BN separately)")
    args = parser.parse_args()

    # ── 1. Locate val image ───────────────────────────────────────────────────

    coco_val = DATASETS_CACHE_DIR / "coco128" / "val"
    images = sorted((coco_val / "images").glob("*.jpg"))
    assert images, f"No images found under {coco_val / 'images'} — run download first"
    assert args.image_idx < len(images), \
        f"image-idx {args.image_idx} out of range (0..{len(images)})"

    img_path = images[args.image_idx]
    print(f"Image [{args.image_idx}]: {img_path.name}")
    orig = cv2.imread(str(img_path))
    print(f"  Orig size: {orig.shape[1]}×{orig.shape[0]}")

    img_t = load_image(img_path)
    print(f"  Preprocessed: shape={list(img_t.shape)}, "
          f"min={img_t.min():.4f}, max={img_t.max():.4f}, mean={img_t.mean():.4f}")

    # ── 2. Load model ─────────────────────────────────────────────────────────

    print(f"\nLoading model from {args.model} ...")
    ckpt = torch.load(args.model, map_location=DEVICE, weights_only=False)
    raw  = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = raw.float().to(DEVICE).eval()

    if not args.no_fuse:
        model.fuse()
        print("Model fused (Conv+BN merged — matches Rust inference path).")
    else:
        print("Model NOT fused (unfused BN — does NOT match Rust inference path).")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── 3. Register per-module forward hooks ──────────────────────────────────

    layer_records: list[dict] = []
    hooks = []

    def make_hook(name: str, mod_type: str):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                s = tensor_stats(out)
                s["name"]  = name
                s["type"]  = mod_type
                layer_records.append(s)
            elif isinstance(out, (list, tuple)):
                for i, o in enumerate(out):
                    if isinstance(o, torch.Tensor):
                        s = tensor_stats(o)
                        s["name"]  = f"{name}[{i}]"
                        s["type"]  = mod_type
                        layer_records.append(s)
        return hook

    for name, module in model.named_modules():
        if name:  # skip the root module itself
            h = module.register_forward_hook(make_hook(name, type(module).__name__))
            hooks.append(h)

    # ── 4. Forward pass ───────────────────────────────────────────────────────

    print(f"\nRunning forward pass ...")
    with torch.no_grad():
        raw_out = model(img_t)
    for h in hooks:
        h.remove()

    print(f"Captured {len(layer_records)} layer outputs.")

    # ── 5. Print per-layer stats ──────────────────────────────────────────────

    sep = "─" * 116
    print(f"\n{sep}")
    print(f"{'#':<5} {'Name':<50} {'Type':<22} {'Shape':<22} "
          f"{'Min':>9} {'Max':>9} {'Mean':>9} {'NaN':>5} {'Inf':>5}")
    print(sep)

    first_bad: str | None = None
    for i, s in enumerate(layer_records):
        flag = " ← BAD" if (s["nans"] > 0 or s["infs"] > 0) else ""
        if (s["nans"] > 0 or s["infs"] > 0) and first_bad is None:
            first_bad = s["name"]
        print(f"{i:<5} {s['name']:<50} {s['type']:<22} {str(s['shape']):<22} "
              f"{s['min']:>9.4f} {s['max']:>9.4f} {s['mean']:>9.4f} "
              f"{s['nans']:>5} {s['infs']:>5}{flag}")

    print(sep)
    if first_bad:
        print(f"FIRST bad output at: {first_bad}")
    else:
        print("All outputs finite — no NaN/Inf detected.")

    # ── 6. Print final model outputs ──────────────────────────────────────────

    print()
    if isinstance(raw_out, (list, tuple)):
        for i, o in enumerate(raw_out):
            if isinstance(o, torch.Tensor):
                s = tensor_stats(o)
                preview = ", ".join(f"{v:.4f}" for v in o.detach().float().flatten()[:8].tolist())
                print(f"Final out[{i}]: shape={s['shape']}, "
                      f"min={s['min']:.4f}, max={s['max']:.4f}, mean={s['mean']:.4f}")
                print(f"  first 8 values: [{preview}, ...]")
    elif isinstance(raw_out, torch.Tensor):
        s = tensor_stats(raw_out)
        preview = ", ".join(f"{v:.4f}" for v in raw_out.detach().float().flatten()[:8].tolist())
        print(f"Final output: shape={s['shape']}, "
              f"min={s['min']:.4f}, max={s['max']:.4f}, mean={s['mean']:.4f}")
        print(f"  first 8 values: [{preview}, ...]")


if __name__ == "__main__":
    main()
