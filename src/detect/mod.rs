/*
 * Copyright 2026 Teenygrad
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


//! Object detection interface.
//!
//! # Example
//!
//! ```rust,no_run
//! use vision_rs::detect::{DetectorConfig, ObjectDetector, RfDetrConfig};
//! use vision_rs::models::detr::rfdetr::rfdetr::RfDetrVariant;
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let config = DetectorConfig::RfDetr(RfDetrConfig::new(
//!     RfDetrVariant::S,
//!     "weights/rfdetr_s.bin",
//!     vec!["car".into(), "truck".into(), "person".into()],
//! ));
//!
//! let detector = ObjectDetector::new(config)?;
//! let image_bytes = std::fs::read("frame.jpg")?;
//! let detections = detector.detect(&image_bytes).await?;
//!
//! for d in &detections {
//!     println!("{} {:.2} {:?}", d.class, d.confidence, d.bbox);
//! }
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;

use crate::models::detr::rfdetr::rfdetr::RfDetrVariant;
use crate::models::yolo::yolo26::Yolo26Variant;

// ── Detection result ──────────────────────────────────────────────────────────

/// A single detected object.
#[derive(Clone, Debug)]
pub struct Detection {
    /// Bounding box as `[cx, cy, w, h]` normalised to `[0, 1]`.
    pub bbox: [f32; 4],
    /// Class label resolved from the detector's `class_names` list.
    pub class: String,
    /// Confidence score in `[0, 1]`.
    pub confidence: f32,
}

// ── YOLO26 config ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Yolo26DetectorConfig {
    pub variant: Yolo26Variant,
    pub weights: PathBuf,
    pub class_names: Vec<String>,
    /// Minimum confidence to retain a detection. Default: `0.25`.
    pub conf_threshold: f32,
    /// IoU threshold for non-maximum suppression. Default: `0.45`.
    pub nms_iou_threshold: f32,
    /// Square input resolution the model was trained at (e.g. `640`).
    pub img_size: usize,
}

impl Yolo26DetectorConfig {
    pub fn new(
        variant: Yolo26Variant,
        weights: impl Into<PathBuf>,
        class_names: Vec<String>,
    ) -> Self {
        Self {
            variant,
            weights: weights.into(),
            class_names,
            conf_threshold: 0.25,
            nms_iou_threshold: 0.45,
            img_size: 640,
        }
    }
}

// ── RF-DETR config ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct RfDetrConfig {
    pub variant: RfDetrVariant,
    pub weights: PathBuf,
    pub class_names: Vec<String>,
    /// Minimum confidence to retain a detection. Default: `0.5`.
    pub conf_threshold: f32,
    pub img_h: usize,
    pub img_w: usize,
}

impl RfDetrConfig {
    pub fn new(
        variant: RfDetrVariant,
        weights: impl Into<PathBuf>,
        class_names: Vec<String>,
    ) -> Self {
        Self {
            variant,
            weights: weights.into(),
            class_names,
            conf_threshold: 0.5,
            img_h: 560,
            img_w: 560,
        }
    }
}

// ── Top-level config enum ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum DetectorConfig {
    Yolo26(Yolo26DetectorConfig),
    RfDetr(RfDetrConfig),
}

// ── ObjectDetector ────────────────────────────────────────────────────────────

pub struct ObjectDetector {
    inner: DetectorInner,
}

impl ObjectDetector {
    /// Build a detector from the given config.
    pub fn new(config: DetectorConfig) -> anyhow::Result<Self> {
        let inner = match config {
            DetectorConfig::Yolo26(cfg) => DetectorInner::Yolo26(Yolo26Detector { config: cfg }),
            DetectorConfig::RfDetr(cfg) => DetectorInner::RfDetr(RfDetrDetector { config: cfg }),
        };
        Ok(Self { inner })
    }

    /// Run inference on raw JPEG or PNG `image_bytes`.
    ///
    /// Returns every detection that clears the config's `conf_threshold`,
    /// after non-maximum suppression (YOLO26) or query filtering (RF-DETR).
    pub async fn detect(&self, image_bytes: &[u8]) -> anyhow::Result<Vec<Detection>> {
        match &self.inner {
            DetectorInner::Yolo26(d) => d.detect(image_bytes).await,
            DetectorInner::RfDetr(d) => d.detect(image_bytes).await,
        }
    }
}

// ── Private per-model implementations ────────────────────────────────────────

enum DetectorInner {
    Yolo26(Yolo26Detector),
    RfDetr(RfDetrDetector),
}

struct Yolo26Detector {
    config: Yolo26DetectorConfig,
}

struct RfDetrDetector {
    config: RfDetrConfig,
}

impl Yolo26Detector {
    async fn detect(&self, _image_bytes: &[u8]) -> anyhow::Result<Vec<Detection>> {
        let _ = &self.config;
        todo!("YOLO26 inference")
    }
}

impl RfDetrDetector {
    async fn detect(&self, _image_bytes: &[u8]) -> anyhow::Result<Vec<Detection>> {
        let _ = &self.config;
        todo!("RF-DETR inference")
    }
}
