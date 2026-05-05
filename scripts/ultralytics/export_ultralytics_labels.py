#!/usr/bin/env python3
"""
Convert vision-rs coco128 labels.toml files into Ultralytics YOLO label files.

Input layout (existing):
  <dataset_root>/
    train/
      images/*.jpg
      labels.toml
    val/
      images/*.jpg
      labels.toml

Output layout (created):
  <dataset_root>/
    train/
      labels/*.txt
    val/
      labels/*.txt

Usage:
  python scripts/export_ultralytics_labels.py
  python scripts/export_ultralytics_labels.py --dataset-root /mnt/data1/datasets/cache/coco128
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="/mnt/data1/datasets/cache/coco128",
        help="Root directory containing train/ and val/ (default: %(default)s).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to convert (default: train val).",
    )
    return parser.parse_args()


def convert_split(dataset_root: Path, split: str) -> tuple[int, int]:
    split_dir = dataset_root / split
    labels_toml = split_dir / "labels.toml"
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not labels_toml.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_toml}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    data = tomllib.loads(labels_toml.read_text(encoding="utf-8"))
    images = data.get("images", [])
    labels_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_images = 0

    for image_item in images:
        file_name = image_item["file"]
        image_path = images_dir / file_name
        if not image_path.exists():
            missing_images += 1
            continue

        stem = Path(file_name).stem
        out_path = labels_dir / f"{stem}.txt"
        annotations = image_item.get("annotations", [])

        lines = []
        for ann in annotations:
            cls = int(ann["class_id"])
            cx, cy, bw, bh = ann["bbox"]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        written += 1

    return written, missing_images


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    print(f"Converting labels under: {dataset_root}")
    for split in args.splits:
        written, missing = convert_split(dataset_root, split)
        print(f"[{split}] wrote {written} YOLO label files")
        if missing:
            print(f"[{split}] skipped {missing} entries with missing images")


if __name__ == "__main__":
    main()
