# Benchmarking & Profiling

## Throughput/latency (`yolo26 bench`)

```bash
source .env
cargo build --release --example yolo26 --features cuda
./target/release/examples/yolo26 bench \
  --model ultralytics/yolo26n \
  --dataset assets/datasets/coco128.toml \
  --skip-map \
  --warmup 10 \
  --runs 100
```

`--skip-map` skips the mAP@0.5 accuracy check (adds ~30s if included).
`--warmup`/`--runs` control how many iterations are discarded vs. timed.

## Per-kernel profiling with `nsys`

The bench command wraps its batch=1 timed loop with
`cudaProfilerStart`/`Stop`, so `--capture-range=cudaProfilerApi` records
only the timed region, excluding warmup and kernel compilation:

```bash
source .env
nsys profile \
  --capture-range=cudaProfilerApi \
  --cuda-graph-trace=node \
  --output=/tmp/yolo26_bench \
  --force-overwrite=true \
  --stats=true \
  --kill=none \
  ./target/release/examples/yolo26 bench \
    --model ultralytics/yolo26n \
    --dataset assets/datasets/coco128.toml \
    --skip-map \
    --warmup 10 \
    --runs 100
```

- `--cuda-graph-trace=node` gets individual kernel stats inside CUDA
  graphs, which would otherwise be opaque to `nsys`.
- `--kill=none` stops `nsys` from `SIGTERM`-ing the process after the
  capture range ends — without it, the bench command's final results row
  never gets printed.
- The report lands at `/tmp/yolo26_bench.nsys-rep` (open in the Nsight
  Systems GUI). `--stats=true` also prints `cuda_gpu_kern_sum`,
  `cuda_api_sum`, and `cuda_gpu_mem_time_sum` tables directly to stdout.

## Comparing against a TensorRT baseline

`bench.py` runs the same benchmark across PyTorch, ONNX Runtime, and
TensorRT FP32, using the `ultralytics` Python package:

```bash
source .env
python3 bench.py
```

The TensorRT engine export takes ~25s on first run; the compiled `.engine`
file is cached at `$MODELS_CACHE_DIR/ultralytics/yolo26n/yolo26n.engine`
for subsequent runs.
