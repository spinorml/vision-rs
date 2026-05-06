#!/usr/bin/env python3
"""
Convert Ultralytics .pt model files to safetensors format.
Handles Ultralytics-specific .pt structure.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

# torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])


def convert_ultralytics_to_safetensors(pt_path: str, output_path: str = None):
    pt_file = Path(pt_path)
    if not pt_file.exists():
        print(f"Error: {pt_path} not found")
        return False

    # Default output path
    if output_path is None:
        output_path = pt_file.with_suffix(".safetensors")

    print(f"Loading {pt_file.name}...")

    try:
        # Load the .pt file
        checkpoint = torch.load(pt_file, map_location="cpu", weights_only=False)

        # Ultralytics .pt files have specific structure
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                # Full checkpoint with model object
                model = checkpoint["model"]
                if hasattr(model, "state_dict"):
                    state_dict = model.state_dict()
                else:
                    state_dict = model
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        elif hasattr(checkpoint, "state_dict"):
            # Model object directly
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint

        # Print some info
        print(f"Found {len(state_dict)} tensors")

        # Convert all tensors to float32
        # (some models use float16 or bfloat16 internally)
        converted = {}
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                converted[key] = tensor.float().contiguous()
            else:
                print(f"Skipping non-tensor key: {key} (type: {type(tensor)})")

        # Save as safetensors
        print(f"Saving to {output_path}...")
        save_file(converted, str(output_path))
        print(f"✓ Done — {len(converted)} tensors saved")

        # Print layer names for debugging
        print("\nFirst 10 layer names:")
        for i, key in enumerate(converted.keys()):
            if i >= 10:
                print("  ...")
                break
            print(f"  {key}: {converted[key].shape}")

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def convert_folder(folder_path: str, output_folder: str = None):
    folder = Path(folder_path)
    output = Path(output_folder) if output_folder else folder
    output.mkdir(parents=True, exist_ok=True)

    pt_files = list(folder.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {folder_path}")
        return

    print(f"Found {len(pt_files)} .pt files\n")
    for pt_file in pt_files:
        out_file = output / pt_file.with_suffix(".safetensors").name
        convert_ultralytics_to_safetensors(str(pt_file), str(out_file))
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file:  python convert.py model.pt [output.safetensors]")
        print("  Folder:       python convert.py ./weights/ [./output/]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if input_path.is_dir():
        convert_folder(str(input_path), output_path)
    else:
        convert_ultralytics_to_safetensors(str(input_path), output_path)
