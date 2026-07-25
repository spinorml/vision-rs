#!/usr/bin/env python3
"""
Generate a coco128-style subset from local COCO-2017 zip files.

Samples --n images independently from train2017 and val2017, copies them to:
  <dataset_dir>/coco128/{train,val}/images/

and writes a single labels.toml per split in YOLO convention
(bounding boxes as normalised [cx, cy, w, h]).

Also updates the [local] section of assets/models/coco128.toml.

Usage (from the vision-rs repo root):
    python scripts/generate_coco128.py
    python scripts/generate_coco128.py --n 256
    python scripts/generate_coco128.py --n 128 --seed 7
    python scripts/generate_coco128.py --dataset-dir /mnt/data1/datasets/coco-2017 --n 128
"""

import argparse
import json
import random
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# COCO 80 detection classes in standard YOLO order (0-based index).
# ---------------------------------------------------------------------------
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

_NAME_TO_IDX = {name: idx for idx, name in enumerate(COCO_CLASSES)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="/mnt/data1/datasets/coco-2017",
                   help="Directory containing the COCO-2017 zip files (default: %(default)s).")
    p.add_argument("--n", type=int, default=128,
                   help="Number of images to sample per split (default: %(default)s).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: %(default)s).")
    p.add_argument("--metadata-toml", default=None,
                   help="Path to assets/models/coco128.toml to update (auto-detected if omitted).")
    return p.parse_args()


def load_instances(annotations_zip: Path, split: str) -> dict:
    """Read instances_{split}2017.json from the annotations zip into memory."""
    entry = f"annotations/instances_{split}2017.json"
    print(f"  loading {entry} ...")
    with zipfile.ZipFile(annotations_zip) as zf:
        with zf.open(entry) as fh:
            return json.load(fh)


def build_category_map(categories: list) -> dict:
    """Map COCO category_id (non-contiguous) → 0-based YOLO class index."""
    return {
        cat["id"]: _NAME_TO_IDX[cat["name"]]
        for cat in categories
        if cat["name"] in _NAME_TO_IDX
    }


def sample_images(data: dict, n: int, rng: random.Random) -> list[dict]:
    """Return up to n randomly sampled images that have at least one annotation."""
    annotated = {ann["image_id"] for ann in data["annotations"]}
    candidates = [img for img in data["images"] if img["id"] in annotated]
    return rng.sample(candidates, min(n, len(candidates)))


def extract_images(selected: list[dict], src_zip: Path, out_dir: Path) -> None:
    """Copy selected images out of the zip into out_dir."""
    needed = {img["file_name"] for img in selected}
    done = 0
    print(f"  extracting {len(needed)} images from {src_zip.name} ...")
    with zipfile.ZipFile(src_zip) as zf:
        for entry in zf.infolist():
            fname = Path(entry.filename).name
            if fname not in needed:
                continue
            (out_dir / fname).write_bytes(zf.read(entry.filename))
            done += 1
            if done % 64 == 0:
                print(f"    {done}/{len(needed)}")
    print(f"  extracted {done} images.")


def bbox_to_yolo(x: float, y: float, w: float, h: float,
                 img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert COCO [x_min, y_min, w, h] pixels → YOLO [cx, cy, w, h] normalised."""
    return (
        round((x + w / 2) / img_w, 6),
        round((y + h / 2) / img_h, 6),
        round(w / img_w, 6),
        round(h / img_h, 6),
    )


# ---------------------------------------------------------------------------
# TOML writers
# ---------------------------------------------------------------------------

def write_labels_toml(
    selected: list[dict],
    data: dict,
    cat_map: dict,
    split: str,
    seed: int,
    src_zip: Path,
    out_path: Path,
) -> None:
    """Write labels.toml for one split."""
    ann_by_image: dict[int, list] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if ann["category_id"] not in cat_map:
            continue
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    lines: list[str] = [
        "# Generated by scripts/generate_coco128.py — do not edit by hand.",
        f'# source: {src_zip}',
        "",
        "[dataset]",
        f'split      = "{split}"',
        f"n_images   = {len(selected)}",
        f"seed       = {seed}",
        f'source_zip = "{src_zip}"',
        "",
        "[classes]",
        'names = [',
    ]
    for i in range(0, len(COCO_CLASSES), 5):
        chunk = COCO_CLASSES[i:i + 5]
        sep = "," if i + 5 < len(COCO_CLASSES) else ""
        lines.append('  "' + '", "'.join(chunk) + '"' + sep)
    lines.append("]")
    lines.append("")

    for img in selected:
        img_id = img["id"]
        iw, ih = img["width"], img["height"]

        lines += [
            "[[images]]",
            f'id     = {img_id}',
            f'file   = "{img["file_name"]}"',
            f"width  = {iw}",
            f"height = {ih}",
        ]

        for ann in ann_by_image.get(img_id, []):
            cx, cy, bw, bh = bbox_to_yolo(*ann["bbox"], iw, ih)
            lines += [
                "",
                "  [[images.annotations]]",
                f"  class_id = {cat_map[ann['category_id']]}",
                f"  bbox     = [{cx}, {cy}, {bw}, {bh}]",
            ]

        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {out_path}  ({len(selected)} images)")


def update_metadata_toml(metadata_path: Path, n: int, seed: int, coco128_dir: Path) -> None:
    """Replace the [local] section in coco128.toml with up-to-date paths."""
    if not metadata_path.exists():
        print(f"  metadata TOML not found at {metadata_path}, skipping.")
        return

    lines = metadata_path.read_text(encoding="utf-8").splitlines()

    # Strip any existing [local] section (everything from [local] to the next
    # top-level section or end of file).
    filtered, in_local = [], False
    for line in lines:
        if line.strip() == "[local]":
            in_local = True
            continue
        if in_local and line.startswith("["):
            in_local = False
        if not in_local:
            filtered.append(line)

    # Remove trailing blank lines before appending the new section.
    while filtered and filtered[-1].strip() == "":
        filtered.pop()

    filtered += [
        "",
        "[local]",
        f'path  = "{coco128_dir}"',
        f"n     = {n}",
        f"seed  = {seed}",
        f'train = "{coco128_dir / "train" / "labels.toml"}"',
        f'val   = "{coco128_dir / "val"   / "labels.toml"}"',
        "",
    ]

    metadata_path.write_text("\n".join(filtered), encoding="utf-8")
    print(f"  updated {metadata_path}")


def find_metadata_toml() -> Path:
    """Resolve assets/models/coco128.toml relative to this script."""
    return Path(__file__).resolve().parent.parent / "assets" / "models" / "coco128.toml"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    dataset_dir     = Path(args.dataset_dir)
    n, seed         = args.n, args.seed
    rng             = random.Random(seed)
    annotations_zip = dataset_dir / "annotations_trainval2017.zip"
    coco128_dir     = dataset_dir / "coco128"
    metadata_path   = Path(args.metadata_toml) if args.metadata_toml else find_metadata_toml()

    for path in (annotations_zip,):
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")

    for split, zip_name in [("train", "train2017.zip"), ("val", "val2017.zip")]:
        src_zip   = dataset_dir / zip_name
        images_dir = coco128_dir / split / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n── {split} ──────────────────────────────────────────────")

        if not src_zip.exists():
            raise FileNotFoundError(f"Expected file not found: {src_zip}")

        data     = load_instances(annotations_zip, split)
        cat_map  = build_category_map(data["categories"])
        selected = sample_images(data, n, rng)

        extract_images(selected, src_zip, images_dir)

        write_labels_toml(
            selected, data, cat_map, split, seed, src_zip,
            coco128_dir / split / "labels.toml",
        )

    print(f"\n── metadata ────────────────────────────────────────────────")
    update_metadata_toml(metadata_path, n, seed, coco128_dir)

    print(f"\nDone.  Dataset at {coco128_dir}")


if __name__ == "__main__":
    main()
