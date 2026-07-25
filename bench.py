from pathlib import Path
from ultralytics.utils.benchmarks import benchmark

REPO_ROOT = Path(__file__).resolve().parent
MODEL = "/mnt/data1/models/cache/ultralytics/yolo26n/yolo26n.pt"
DATA = str(REPO_ROOT / "scripts" / "coco128_local.yaml")

# Benchmark all formats on GPU (batch=1, 128 val images)
benchmark(model=MODEL, data=DATA, imgsz=640, half=False, device=0)
