# Your First Detection

The `vision_rs::detect` module is the entry point for running inference. It
wraps model-specific configuration and forward passes behind a single
`ObjectDetector`/`DetectorConfig` pair, so callers don't need to touch the
model internals for simple inference.

```rust,no_run
use vision_rs::detect::{DetectorConfig, ObjectDetector, Yolo26DetectorConfig};
use vision_rs::models::yolo::yolo26::Yolo26Variant;

# #[tokio::main]
# async fn main() -> anyhow::Result<()> {
let config = DetectorConfig::Yolo26(Yolo26DetectorConfig::new(
    Yolo26Variant::N,
    "weights/yolo26n.bin",
    vec!["car".into(), "truck".into(), "person".into()],
));

let detector = ObjectDetector::new(config)?;
let image_bytes = std::fs::read("frame.jpg")?;
let detections = detector.detect(&image_bytes).await?;

for d in &detections {
    println!("{} {:.2} {:?}", d.class, d.confidence, d.bbox);
}
# Ok(())
# }
```

## Walking through it

1. **Pick a variant.** `Yolo26Variant` has five sizes — `N` (nano) through
   `XL` — trading accuracy for speed/memory. See
   [The YOLO26 Model](../core-concepts/yolo26-architecture.md) for how they
   differ.
2. **Build a config.** `Yolo26DetectorConfig::new` takes the variant, a
   weights file path, and the class label list (must match the model's
   training classes, in class-index order). It fills in sensible defaults:
   `conf_threshold: 0.25`, `nms_iou_threshold: 0.45`, `img_size: 640`. Adjust
   these fields directly on the returned config if your use case needs
   different thresholds.
3. **Construct the detector.** `ObjectDetector::new` takes ownership of the
   config and loads the model.
4. **Run inference.** `detect()` takes raw JPEG or PNG bytes and returns
   every [`Detection`](https://docs.rs/vision-rs/latest/vision_rs/detect/struct.Detection.html)
   that clears `conf_threshold`, after non-maximum suppression. Each
   `Detection` has a `bbox: [cx, cy, w, h]` (normalised to `[0, 1]`), a
   resolved `class` label, and a `confidence` score.

## Getting a model

The `yolo26` example binary (`examples/yolo26.rs`) has a `download`
subcommand for fetching pretrained weights and a dataset config, and a
`bench` subcommand for a throughput/latency smoke test:

```bash
source .env
cargo build --release --example yolo26 --features cuda
./target/release/examples/yolo26 download --dataset assets/datasets/coco128.toml
./target/release/examples/yolo26 bench \
  --model ultralytics/yolo26n \
  --dataset assets/datasets/coco128.toml \
  --skip-map --warmup 10 --runs 100
```

See [Benchmarking & Profiling](../kernels-and-performance/benchmarking.md)
for the full set of `yolo26` subcommands and profiling workflows.
