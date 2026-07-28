# The Detection API

`vision_rs::detect` is the model-agnostic entry point described in
[Your First Detection](../getting-started/first-detection.md). This page
covers its shape in more depth.

## `DetectorConfig`

```rust,ignore
pub enum DetectorConfig {
    Yolo26(Yolo26DetectorConfig),
}
```

`DetectorConfig` is a tagged union over model families. Today it has one
variant, `Yolo26`; as vision-rs grows additional model families, they'll be
added as new variants here rather than as separate top-level detector types
— callers write against `ObjectDetector`/`DetectorConfig` regardless of
which model backs it.

## `Yolo26DetectorConfig`

```rust,ignore
pub struct Yolo26DetectorConfig {
    pub variant: Yolo26Variant,
    pub weights: PathBuf,
    pub class_names: Vec<String>,
    pub conf_threshold: f32,      // default 0.25
    pub nms_iou_threshold: f32,   // default 0.45
    pub img_size: usize,          // default 640
}
```

`Yolo26DetectorConfig::new(variant, weights, class_names)` fills in the
three threshold/size fields with the defaults above; mutate them directly on
the returned struct if you need different behavior (e.g. a lower
`conf_threshold` for a high-recall use case, or a different `img_size` if
your weights were trained at a non-standard resolution).

## `ObjectDetector`

```rust,ignore
pub struct ObjectDetector { /* ... */ }

impl ObjectDetector {
    pub fn new(config: DetectorConfig) -> anyhow::Result<Self>;
    pub async fn detect(&self, image_bytes: &[u8]) -> anyhow::Result<Vec<Detection>>;
}
```

`new` dispatches on the config variant to build the right underlying model.
`detect` is `async` — model loading/inference may involve device transfers
and kernel launches — and takes raw encoded image bytes (JPEG or PNG)
rather than a pre-decoded tensor, so callers don't need a separate image
decoding dependency for the common case.

## `Detection`

```rust,ignore
pub struct Detection {
    pub bbox: [f32; 4],   // [cx, cy, w, h], normalised to [0, 1]
    pub class: String,    // resolved from the config's class_names
    pub confidence: f32,  // in [0, 1]
}
```

Note that `bbox` is **normalised**, not pixel coordinates — multiply by the
original image's width/height to get pixel-space boxes. `class` is already
resolved to a string label (not a raw class index), using the
`class_names` list passed into the config.
