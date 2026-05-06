#!/usr/bin/env python3
"""
Convert an Ultralytics .pt model file to safetensors format.

Usage:
    python3 convert_model.py <input.pt> <output.safetensors>
"""

import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: convert_model.py <input.pt> <output.safetensors>", file=sys.stderr
        )
        sys.exit(1)

    pt_path = Path(sys.argv[1])
    st_path = Path(sys.argv[2])

    if not pt_path.exists():
        print(f"Error: {pt_path} not found", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        print(f"Error: missing dependency — {e}", file=sys.stderr)
        print(
            "Install with: pip install torch safetensors ultralytics", file=sys.stderr
        )
        sys.exit(1)

    print(f"Loading {pt_path.name} ...")
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            model_obj = checkpoint["model"]
            state_dict = (
                model_obj.state_dict()
                if hasattr(model_obj, "state_dict")
                else model_obj
            )
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    elif hasattr(checkpoint, "state_dict"):
        state_dict = checkpoint.state_dict()
    else:
        state_dict = checkpoint

    print(f"Found {len(state_dict)} tensors")

    converted = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            converted[key] = tensor.float().contiguous()
        else:
            print(f"  Skipping non-tensor key: {key} ({type(tensor).__name__})")

    st_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {st_path} ...")
    save_file(converted, str(st_path))
    print(f"Done — {len(converted)} tensors written")


if __name__ == "__main__":
    main()
