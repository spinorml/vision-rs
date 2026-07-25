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


//! RF-DETR example: view, verify, and train.
//!
//! Usage:
//!   # View training images with GT boxes
//!   cargo run --example rfdetr -- view --dataset assets/datasets/coco128.toml
//!
//!   # View with live inference overlay (CUDA)
//!   cargo run --example rfdetr -- view --dataset assets/datasets/coco128.toml \
//!       --model assets/models/rfdetr/rfdetr_b.toml
//!
//!   # Verify mAP against a pre-trained safetensors checkpoint
//!   cargo run --example rfdetr -- verify \
//!       --model assets/models/rfdetr/rfdetr_b.toml \
//!       --dataset assets/datasets/coco128.toml
//!
//!   # Train from scratch
//!   cargo run --example rfdetr -- train \
//!       --dataset assets/datasets/coco128.toml \
//!       --variant b --batch-size 2 --epochs 10 \
//!       --checkpoint /tmp/rfdetr_ckpt

use std::collections::HashMap;
use std::path::{Path, PathBuf};

type InferFn = Box<dyn FnMut(&Path) -> anyhow::Result<Vec<(usize, f32, [f32; 4])>>>;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dotenv::dotenv;
use eframe::egui;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "RF-DETR: view, verify, and train on a vision-rs dataset")]
struct Args {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// View training images with bounding boxes
    View {
        #[arg(short, long)]
        dataset: PathBuf,
        /// Optional model TOML (e.g. assets/models/rfdetr/rfdetr_b.toml) for inference overlay
        #[arg(long)]
        model: Option<PathBuf>,
        /// Input resolution (square)
        #[arg(long, default_value_t = 560)]
        img_size: usize,
    },
    /// Verify mAP against a pre-trained model on the dataset validation split
    Verify {
        /// Path to model config TOML
        #[arg(long)]
        model: PathBuf,
        /// Path to dataset config TOML
        #[arg(short, long)]
        dataset: PathBuf,
        /// Input resolution (square)
        #[arg(long, default_value_t = 560)]
        img_size: usize,
        /// Batch size for inference
        #[arg(short = 'b', long, default_value_t = 4)]
        batch_size: usize,
    },
    /// Benchmark throughput and latency across batch sizes (CUDA graphs)
    Bench {
        /// Path to model config TOML
        #[arg(long)]
        model: PathBuf,
        /// Path to dataset config TOML
        #[arg(short, long)]
        dataset: PathBuf,
        /// Input resolution (square)
        #[arg(long, default_value_t = 560)]
        img_size: usize,
        /// Number of warmup iterations per batch size
        #[arg(long, default_value_t = 10)]
        warmup: usize,
        /// Number of timed iterations per batch size
        #[arg(long, default_value_t = 100)]
        runs: usize,
        /// Skip mAP evaluation (faster startup)
        #[arg(long)]
        skip_map: bool,
    },
    /// Train RF-DETR on a dataset
    Train {
        /// Path to dataset config TOML
        #[arg(short, long)]
        dataset: PathBuf,
        /// Input resolution (square)
        #[arg(long, default_value_t = 560)]
        img_size: usize,
        /// Batch size
        #[arg(short = 'b', long, default_value_t = 2)]
        batch_size: usize,
        /// Number of epochs
        #[arg(short = 'e', long, default_value_t = 10)]
        epochs: usize,
        /// Learning rate
        #[arg(long, default_value_t = 1e-4)]
        lr: f64,
        /// Directory to save and resume checkpoints
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Model variant: s | b
        #[arg(long, default_value = "b")]
        variant: String,
        /// Number of classes (inferred from dataset if omitted)
        #[arg(long)]
        nc: Option<usize>,
    },
}

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ModelConfig {
    model: ModelMeta,
    #[allow(dead_code)]
    download: ModelDownload,
    #[serde(default)]
    weights: ModelWeights,
}

#[derive(Deserialize)]
struct ModelMeta {
    #[allow(dead_code)]
    name: String,
    variant: String,
    nc: usize,
}

#[derive(Deserialize)]
struct ModelDownload {
    #[allow(dead_code)]
    url: String,
    #[allow(dead_code)]
    filename: String,
}

#[derive(Deserialize, Default)]
struct ModelWeights {
    #[serde(default)]
    mapping: HashMap<String, String>,
}

#[derive(Deserialize)]
struct DatasetConfig {
    dataset: DatasetMeta,
    #[serde(default)]
    classes: ClassesMeta,
}

#[derive(Deserialize)]
struct DatasetMeta {
    name: String,
    #[allow(dead_code)]
    url: String,
}

#[derive(Deserialize)]
struct LabelsFile {
    classes: ClassesMeta,
    #[serde(default)]
    images: Vec<ImageEntry>,
}

#[derive(Deserialize, Default)]
struct ClassesMeta {
    names: Vec<String>,
}

#[derive(Deserialize, Clone)]
struct ImageEntry {
    file: String,
    #[serde(default)]
    annotations: Vec<BBox>,
}

#[derive(Deserialize, Clone)]
struct BBox {
    class_id: usize,
    bbox: [f32; 4], // [cx, cy, w, h] normalised [0, 1]
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    dotenv().ok();
    let args = Args::parse();

    match args.command {
        Cmd::View {
            dataset,
            model,
            img_size,
        } => run_view(dataset, model, img_size),
        Cmd::Verify {
            model,
            dataset,
            img_size,
            batch_size,
        } => run_verify(model, dataset, img_size, batch_size),
        Cmd::Bench {
            model,
            dataset,
            img_size,
            warmup,
            runs,
            skip_map,
        } => run_bench_rfdetr(model, dataset, img_size, warmup, runs, skip_map),
        Cmd::Train {
            dataset,
            img_size,
            batch_size,
            epochs,
            lr,
            checkpoint,
            variant,
            nc,
        } => run_train(
            dataset, img_size, batch_size, epochs, lr as f32, checkpoint, variant, nc,
        ),
    }
}

// ---------------------------------------------------------------------------
// View command
// ---------------------------------------------------------------------------

fn run_view(dataset: PathBuf, model_path: Option<PathBuf>, img_size: usize) -> Result<()> {
    let config_text = std::fs::read_to_string(&dataset)
        .with_context(|| format!("reading dataset config {:?}", dataset))?;
    let config: DatasetConfig =
        toml::from_str(&config_text).context("parsing dataset config TOML")?;

    let cache_dir: PathBuf = std::env::var("DATASETS_CACHE_DIR")
        .context("DATASETS_CACHE_DIR not set — add it to .env")?
        .into();

    let dataset_dir = cache_dir.join(&config.dataset.name);
    let labels_path = dataset_dir.join("train").join("labels.toml");
    let images_dir = dataset_dir.join("train").join("images");

    let labels_text = std::fs::read_to_string(&labels_path)
        .with_context(|| format!("reading {:?}", labels_path))?;
    let labels: LabelsFile = toml::from_str(&labels_text).context("parsing labels.toml")?;

    if labels.images.is_empty() {
        anyhow::bail!("no images in {:?}", labels_path);
    }

    #[cfg(feature = "cuda")]
    let infer_fn: Option<InferFn> = match model_path {
        Some(ref p) => Some(build_view_infer_fn(p, img_size)?),
        None => None,
    };
    #[cfg(not(feature = "cuda"))]
    let infer_fn: Option<InferFn> = {
        if model_path.is_some() {
            eprintln!("Warning: --model requires the 'cuda' feature; inference overlay disabled.");
        }
        None
    };

    let title = format!(
        "RF-DETR · {} — train ({} images)",
        config.dataset.name,
        labels.images.len()
    );

    let app = ViewApp::new(
        labels.images,
        labels.classes.names,
        images_dir,
        infer_fn,
        img_size,
    );
    eframe::run_native(
        &title,
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
            ..Default::default()
        },
        Box::new(|_cc| Ok(Box::new(app))),
    )
    .map_err(|e| anyhow::anyhow!("eframe: {e}"))
}

// ---------------------------------------------------------------------------
// Viewer
// ---------------------------------------------------------------------------

struct ViewApp {
    images_dir: PathBuf,
    entries: Vec<ImageEntry>,
    classes: Vec<String>,
    idx: usize,
    jump_buf: String,
    texture: Option<egui::TextureHandle>,
    loaded_idx: Option<usize>,
    infer_fn: Option<InferFn>,
    detections: Vec<(usize, f32, [f32; 4])>,
}

impl ViewApp {
    fn new(
        entries: Vec<ImageEntry>,
        classes: Vec<String>,
        images_dir: PathBuf,
        infer_fn: Option<InferFn>,
        _img_size: usize,
    ) -> Self {
        Self {
            images_dir,
            entries,
            classes,
            idx: 0,
            jump_buf: String::new(),
            texture: None,
            loaded_idx: None,
            infer_fn,
            detections: Vec::new(),
        }
    }

    fn prev(&mut self) {
        if self.idx > 0 {
            self.idx -= 1;
        }
    }
    fn next(&mut self) {
        if self.idx + 1 < self.entries.len() {
            self.idx += 1;
        }
    }
    fn jump(&mut self, one_based: usize) {
        self.idx = one_based
            .saturating_sub(1)
            .min(self.entries.len().saturating_sub(1));
    }

    fn load_texture(&mut self, ctx: &egui::Context) {
        let path = self.images_dir.join(&self.entries[self.idx].file);
        self.loaded_idx = Some(self.idx);
        self.detections.clear();

        match image::open(&path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let ci = egui::ColorImage::from_rgba_unmultiplied(size, &rgba);
                self.texture =
                    Some(ctx.load_texture("dataset-image", ci, egui::TextureOptions::LINEAR));
            }
            Err(e) => {
                eprintln!("failed to load {:?}: {e}", path);
                self.texture = None;
            }
        }

        if let Some(infer) = self.infer_fn.as_mut() {
            match infer(&path) {
                Ok(dets) => self.detections = dets,
                Err(e) => eprintln!("inference failed for {:?}: {e}", path),
            }
        }
    }
}

impl eframe::App for ViewApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.input(|i| {
            if i.key_pressed(egui::Key::ArrowLeft) {
                self.prev();
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                self.next();
            }
        });

        if self.loaded_idx != Some(self.idx) {
            self.load_texture(ctx);
        }

        egui::TopBottomPanel::bottom("nav").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("⬅ Prev").clicked() {
                    self.prev();
                }
                ui.label(format!("{} / {}", self.idx + 1, self.entries.len()));
                if ui.button("Next ➡").clicked() {
                    self.next();
                }

                ui.separator();
                ui.label("Go to:");
                let resp =
                    ui.add(egui::TextEdit::singleline(&mut self.jump_buf).desired_width(56.0));
                if resp.lost_focus() && ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if let Ok(n) = self.jump_buf.trim().parse::<usize>() {
                        self.jump(n);
                    }
                }

                ui.separator();
                let entry = &self.entries[self.idx];
                let det_str = if self.infer_fn.is_some() {
                    format!(" · {} detections (model)", self.detections.len())
                } else {
                    String::new()
                };
                ui.label(format!(
                    "{}  ({} boxes{})",
                    entry.file,
                    entry.annotations.len(),
                    det_str
                ));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(texture) = &self.texture {
                let tex_size = texture.size_vec2();
                let available = ui.available_size();
                let scale = (available.x / tex_size.x).min(available.y / tex_size.y);
                let display = tex_size * scale;
                let (rect, _) = ui.allocate_exact_size(display, egui::Sense::hover());
                let painter = ui.painter();

                painter.image(
                    texture.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );

                // GT boxes (coloured by class)
                for ann in &self.entries[self.idx].annotations.clone() {
                    draw_box(
                        &painter,
                        rect,
                        ann.bbox,
                        class_color(ann.class_id),
                        2.0,
                        self.classes
                            .get(ann.class_id)
                            .map(|s| s.as_str())
                            .unwrap_or("?"),
                        class_color(ann.class_id),
                        false,
                    );
                }

                // Inference detections — black box/label for high contrast
                let det_color = egui::Color32::BLACK;
                let det_bg = egui::Color32::BLACK;
                for &(cls_id, _score, bbox) in &self.detections {
                    draw_box(
                        &painter,
                        rect,
                        bbox,
                        det_color,
                        3.5,
                        self.classes.get(cls_id).map(|s| s.as_str()).unwrap_or("?"),
                        det_bg,
                        true,
                    );
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Loading…");
                });
            }
        });
    }
}

fn draw_box(
    painter: &egui::Painter,
    rect: egui::Rect,
    bbox: [f32; 4],
    stroke_col: egui::Color32,
    stroke_w: f32,
    label: &str,
    label_bg: egui::Color32,
    below: bool,
) {
    let [cx, cy, bw, bh] = bbox;
    let x1 = rect.left() + (cx - bw * 0.5) * rect.width();
    let y1 = rect.top() + (cy - bh * 0.5) * rect.height();
    let x2 = rect.left() + (cx + bw * 0.5) * rect.width();
    let y2 = rect.top() + (cy + bh * 0.5) * rect.height();
    let box_rect = egui::Rect::from_min_max(egui::pos2(x1, y1), egui::pos2(x2, y2));

    painter.rect_stroke(box_rect, 0.0, egui::Stroke::new(stroke_w, stroke_col));

    let galley = painter.layout_no_wrap(
        label.to_string(),
        egui::FontId::proportional(12.0),
        egui::Color32::WHITE,
    );
    let label_size = galley.size() + egui::vec2(4.0, 2.0);
    let label_origin = if below {
        egui::pos2(x1, y1.max(rect.top()))
    } else {
        egui::pos2(x1, (y1 - label_size.y).max(rect.top()))
    };
    let bg = egui::Rect::from_min_size(label_origin, label_size);
    painter.rect_filled(bg, 2.0, label_bg);
    painter.galley(
        label_origin + egui::vec2(2.0, 1.0),
        galley,
        egui::Color32::WHITE,
    );
}

// ---------------------------------------------------------------------------
// Color palette
// ---------------------------------------------------------------------------

fn class_color(class_id: usize) -> egui::Color32 {
    let hue = (class_id as f32 * 137.508) % 360.0;
    let h = hue / 60.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    let (r, g, b) = match h as u32 {
        0 => (1.0_f32, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

// ---------------------------------------------------------------------------
// Verify command
// ---------------------------------------------------------------------------

fn run_verify(
    model_path: PathBuf,
    dataset: PathBuf,
    img_size: usize,
    batch_size: usize,
) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (model_path, dataset, img_size, batch_size);
        anyhow::bail!("verify requires the 'cuda' feature");
    }
    #[cfg(feature = "cuda")]
    {
        use teeny_compiler::compiler::{
            backend::llvm::compiler::LlvmCompiler, target::cuda::Target,
        };
        use teeny_core::{
            graph::{DtypeRepr, SymTensor},
            model::LoweringMode,
        };
        use teeny_cuda::{compiler::graph::CudaGraphCompiler, testing};
        use teeny_kernels::graph::TritonLowering;
        use vision_rs::models::detr::rfdetr::rfdetr::rfdetr;

        // ── 1. Parse model config ──────────────────────────────────────────────

        let model_config: ModelConfig = toml::from_str(
            &std::fs::read_to_string(&model_path)
                .with_context(|| format!("reading model config {:?}", model_path))?,
        )
        .context("parsing model config TOML")?;
        let nc = model_config.model.nc;
        let variant_str = model_config.model.variant.clone();

        // ── 2. Parse dataset config ────────────────────────────────────────────

        let ds_config: DatasetConfig = toml::from_str(
            &std::fs::read_to_string(&dataset)
                .with_context(|| format!("reading dataset config {:?}", dataset))?,
        )
        .context("parsing dataset config TOML")?;

        let datasets_cache: PathBuf = std::env::var("DATASETS_CACHE_DIR")
            .context("DATASETS_CACHE_DIR not set")?
            .into();
        let dataset_dir = datasets_cache.join(&ds_config.dataset.name);
        let val_images_dir = dataset_dir.join("val").join("images");
        let val_labels_dir = dataset_dir.join("val").join("labels");

        anyhow::ensure!(
            val_images_dir.exists(),
            "no val images at {:?} — run download first",
            val_images_dir
        );
        let class_names = ds_config.classes.names.clone();
        let val_entries = load_labels_from_dir(&val_images_dir, &val_labels_dir)?;

        // ── 3. CUDA setup ──────────────────────────────────────────────────────

        let env = testing::setup_cuda_env()?;
        let target = Target::new(env.capability);
        let device = &env.device;

        // ── 4. Parse variant ───────────────────────────────────────────────────

        let variant = parse_variant(&variant_str)?;
        let n_queries = variant.n_queries();
        let neck_dim = variant.neck_dim();

        // ── 5. Compile model (inference mode) ─────────────────────────────────

        println!(
            "Compiling RF-DETR-{} (inference, {}×{}, nc={}) ...",
            variant_str.to_uppercase(),
            img_size,
            img_size,
            nc
        );
        println!("(First run compiles all kernels; subsequent runs use the cache.)");

        let teenyc_path = std::env::var("TEENYC_PATH").unwrap_or_else(|_| "teenyc".to_string());
        let kern_cache =
            std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());

        let (img_sym, _graph_rc) = SymTensor::input(
            DtypeRepr::F32,
            vec![None, Some(3), Some(img_size), Some(img_size)],
        );
        // rfdetr() creates the queries Op::Input node internally;
        // compiled model: input[0]=image, input[1]=queries
        let (class_logits, _box_preds) = rfdetr::<f32>(nc, variant, img_size, img_size)(img_sym);
        let graph_rc = class_logits.graph.clone();
        let optimised = graph_rc.borrow().optimise();

        let compiler = LlvmCompiler::new(teenyc_path, kern_cache)?;
        let graph_cmp = CudaGraphCompiler::new(compiler);
        let lowering = TritonLowering::new();
        let cuda_model =
            graph_cmp.compile_model(&optimised, &lowering, &target, LoweringMode::Inference, false)?;

        // ── 6. Load weights ────────────────────────────────────────────────────

        let models_cache: PathBuf = std::env::var("MODELS_CACHE_DIR")
            .context("MODELS_CACHE_DIR not set")?
            .into();
        let st_path = models_cache
            .join(model_path.file_stem().unwrap_or_default())
            .with_extension("safetensors");

        let mut model = cuda_model.load(device, batch_size)?;
        if st_path.exists() {
            println!("Loading weights from {} ...", st_path.display());
            load_weights_from_safetensors(&mut model, &st_path, &model_config.weights.mapping)?;
        } else {
            println!(
                "Note: no weights found at {}; running with random initialisation.",
                st_path.display()
            );
        }

        // ── 7. mAP evaluation ─────────────────────────────────────────────────

        let terminals = model.terminal_node_indices_sorted_by_size();
        anyhow::ensure!(
            terminals.len() >= 2,
            "expected 2 terminal nodes (class_logits, box_preds), got {}",
            terminals.len()
        );
        // Sorted by size: box_preds has 4 cols, class_logits has nc ≥ 1 cols.
        // RF-DETR: box_preds [B, Nq, 4], class_logits [B, Nq, nc].
        // If nc > 4: class_logits is larger → terminals[1].
        // If nc ≤ 4: ambiguous; we order by dag_idx as tiebreak.
        // The rfdetr() closure adds box_preds last (higher dag_idx).
        let (cls_tidx, box_tidx) = (terminals[1], terminals[0]);

        evaluate_map_rfdetr(
            &mut model,
            device,
            &val_entries,
            &val_images_dir,
            &class_names,
            nc,
            n_queries,
            neck_dim,
            img_size,
            batch_size,
            cls_tidx,
            box_tidx,
        )?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Training command
// ---------------------------------------------------------------------------

fn run_train(
    dataset: PathBuf,
    img_size: usize,
    batch_size: usize,
    epochs: usize,
    lr: f32,
    checkpoint: Option<PathBuf>,
    variant_str: String,
    nc_override: Option<usize>,
) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            dataset,
            img_size,
            batch_size,
            epochs,
            lr,
            checkpoint,
            variant_str,
            nc_override,
        );
        anyhow::bail!("train requires the 'cuda' feature");
    }
    #[cfg(feature = "cuda")]
    {
        use teeny_compiler::compiler::{
            backend::llvm::compiler::LlvmCompiler, driver::cuda::compile_kernel,
            target::cuda::Target,
        };
        use teeny_core::{
            graph::{DtypeRepr, SymTensor},
            model::LoweringMode,
        };
        use teeny_cuda::{
            compiler::graph::CudaGraphCompiler,
            model::{AdamwKernel, TensorRef},
            testing,
        };
        use teeny_kernels::{graph::TritonLowering, nn::optim::adam::AdamwStep};
        use vision_rs::models::detr::rfdetr::{loss::matcher::MatchWeights, rfdetr::rfdetr};

        // ── 1. Load dataset ───────────────────────────────────────────────────

        let config_text =
            std::fs::read_to_string(&dataset).with_context(|| format!("reading {:?}", dataset))?;
        let config: DatasetConfig =
            toml::from_str(&config_text).context("parsing dataset config")?;

        let cache_dir: PathBuf = std::env::var("DATASETS_CACHE_DIR")
            .context("DATASETS_CACHE_DIR not set")?
            .into();
        let dataset_dir = cache_dir.join(&config.dataset.name);
        let labels_path = dataset_dir.join("train").join("labels.toml");
        let images_dir = dataset_dir.join("train").join("images");

        let labels_text = std::fs::read_to_string(&labels_path)
            .with_context(|| format!("reading {:?}", labels_path))?;
        let labels: LabelsFile = toml::from_str(&labels_text).context("parsing labels.toml")?;
        let nc = nc_override.unwrap_or(labels.classes.names.len());
        let entries = labels.images;

        println!("Dataset : {}", config.dataset.name);
        println!("Images  : {}", entries.len());
        println!(
            "Classes : {} ({})",
            nc,
            labels
                .classes
                .names
                .join(", ")
                .chars()
                .take(60)
                .collect::<String>()
        );
        println!();

        // ── 2. Pre-process images ─────────────────────────────────────────────

        println!(
            "Pre-processing {} images at {}×{} ...",
            entries.len(),
            img_size,
            img_size
        );
        let mut all_pixels: Vec<Vec<f32>> = Vec::with_capacity(entries.len());
        let pb = progress_bar(entries.len() as u64);
        for entry in &entries {
            all_pixels.push(preprocess_image(&images_dir.join(&entry.file), img_size)?);
            pb.inc(1);
        }
        pb.finish_and_clear();
        println!("Pre-processing complete.\n");

        // ── 3. CUDA setup ─────────────────────────────────────────────────────

        let env = testing::setup_cuda_env()?;
        let target = Target::new(env.capability);
        let device = &env.device;

        // ── 4. Parse variant ───────────────────────────────────────────────────

        let variant = parse_variant(&variant_str)?;
        let n_queries = variant.n_queries();
        let neck_dim = variant.neck_dim();

        // ── 5. Compile model (training mode) ──────────────────────────────────

        println!(
            "Compiling RF-DETR-{} (training, {}×{}, nc={}, Nq={}) ...",
            variant_str.to_uppercase(),
            img_size,
            img_size,
            nc,
            n_queries
        );
        println!("(First run compiles all kernels; subsequent runs use the cache.)");

        let teenyc_path = std::env::var("TEENYC_PATH").unwrap_or_else(|_| "teenyc".to_string());
        let kern_cache =
            std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());

        let (img_sym, _graph_rc) = SymTensor::input(
            DtypeRepr::F32,
            vec![None, Some(3), Some(img_size), Some(img_size)],
        );
        let (class_logits, _box_preds) = rfdetr::<f32>(nc, variant, img_size, img_size)(img_sym);
        let graph_rc = class_logits.graph.clone();
        let graph = graph_rc.borrow();

        let compiler = LlvmCompiler::new(teenyc_path, kern_cache)?;
        let graph_cmp = CudaGraphCompiler::new(compiler);
        let lowering = TritonLowering::new();
        let cuda_model =
            graph_cmp.compile_model(&graph, &lowering, &target, LoweringMode::Training, false)?;
        drop(graph);
        println!("Compiled {} DAG nodes.\n", cuda_model.dag.len());

        // ── 6. Initialise model parameters ────────────────────────────────────

        let mut model = cuda_model.load(device, batch_size)?;
        let param_info: Vec<(usize, Vec<Vec<usize>>)> = model
            .param_info()
            .map(|(idx, shapes)| (idx, shapes.to_vec()))
            .collect();
        let n_params: usize = param_info
            .iter()
            .flat_map(|(_, s)| s.iter().map(|v| v.iter().product::<usize>()))
            .sum();

        if let Some(ref ckpt) = checkpoint {
            let ckpt_file = ckpt.join("params.bin");
            if ckpt_file.exists() {
                println!("Restoring checkpoint from {} ...", ckpt.display());
                restore_checkpoint(&mut model, &param_info, &ckpt_file)?;
            }
        }

        let mut rng: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos() as u64;
        println!("Initialising {n_params} parameters ...");
        for (node_idx, shapes) in &param_info {
            let n_shapes = shapes.len();
            for (param_idx, shape) in shapes.iter().enumerate() {
                let data = init_param(n_shapes, param_idx, shape, &mut rng);
                model.load_param_f32(*node_idx, param_idx, &data)?;
            }
        }
        println!();

        // ── 7. Object queries buffer (initialised to zeros, updated by AdamW) ─

        let query_data = vec![0.0f32; batch_size * n_queries * neck_dim];

        // Spatial metadata for multi-scale deformable attention (3 feature levels).
        // patch_size=14 is hardcoded in RfDetrConfig; level shapes mirror multi_scale_proj outputs.
        let patch_size = 14usize;
        let h_patches = img_size / patch_size;
        let w_patches = img_size / patch_size;
        let h3 = 2 * h_patches;
        let w3 = 2 * w_patches; // s3: 2× upsample
        let h4 = h_patches;
        let w4 = w_patches; // s4: identity
        let h5 = h_patches / 2;
        let w5 = w_patches / 2; // s5: 2× maxpool
        let ss_data = vec![
            h3 as f32, w3 as f32, h4 as f32, w4 as f32, h5 as f32, w5 as f32,
        ];
        let ls_data = vec![0.0f32, (h3 * w3) as f32, (h3 * w3 + h4 * w4) as f32];

        // ── 8. Compile AdamW kernel ───────────────────────────────────────────

        let adamw_ptx = std::fs::read(compile_kernel(&AdamwStep::new(1024), &target, true)?)?;
        let adamw = AdamwKernel::from_ptx(&adamw_ptx)?;

        // ── 9. Terminal indices ────────────────────────────────────────────────

        let terminals = model.terminal_node_indices_sorted_by_size();
        anyhow::ensure!(
            terminals.len() >= 2,
            "expected 2 terminal nodes, got {}",
            terminals.len()
        );
        // box_preds [B,Nq,4] is smaller; class_logits [B,Nq,nc] is larger (nc≥1).
        let (cls_tidx, box_tidx) = (terminals[1], terminals[0]);

        // ── 10. Training loop ─────────────────────────────────────────────────

        let n_batches = entries.len() / batch_size;
        if n_batches == 0 {
            anyhow::bail!(
                "dataset has {} images but batch_size={} — not enough for one batch",
                entries.len(),
                batch_size
            );
        }

        println!(
            "Training: {} images | batch={} | {n_batches} steps/epoch | {epochs} epochs",
            entries.len(),
            batch_size
        );
        println!("Optimiser: AdamW  lr={lr}  β=(0.9, 0.999)  wd=1e-4");
        println!("Loss: BCE class + L1 box (Hungarian matching, Nq={n_queries} queries)\n");

        let mut indices: Vec<usize> = (0..entries.len()).collect();
        let match_w = MatchWeights::default();

        for epoch in 0..epochs {
            // Fisher-Yates shuffle
            let mut rng: u64 = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos() as u64;
            for i in (1..indices.len()).rev() {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = (rng >> 33) as usize % (i + 1);
                indices.swap(i, j);
            }

            let mut epoch_loss = 0.0f32;

            for batch_idx in 0..n_batches {
                let batch_idxs = &indices[batch_idx * batch_size..(batch_idx + 1) * batch_size];

                // Collate batch
                let mut input_data = Vec::with_capacity(batch_size * 3 * img_size * img_size);
                let mut gt_classes = Vec::<Vec<usize>>::with_capacity(batch_size);
                let mut gt_boxes = Vec::<Vec<f32>>::with_capacity(batch_size);
                for &bi in batch_idxs {
                    input_data.extend_from_slice(&all_pixels[bi]);
                    let entry = &entries[bi];
                    gt_classes.push(entry.annotations.iter().map(|a| a.class_id).collect());
                    gt_boxes.push(entry.annotations.iter().flat_map(|a| a.bbox).collect());
                }

                let img_ref =
                    TensorRef::from_host_f32(&input_data, vec![batch_size, 3, img_size, img_size])?;
                let q_ref =
                    TensorRef::from_host_f32(&query_data, vec![batch_size, n_queries, neck_dim])?;

                // spatial_shapes and level_start must be re-created each iteration because
                // ActivationCache::drop frees all device pointers including input TensorRefs.
                let ss_ref = TensorRef::from_host_f32(&ss_data, vec![3, 2])?;
                let ls_ref = TensorRef::from_host_f32(&ls_data, vec![3])?;

                // Forward
                model.zero_grad();
                let (_, cache) =
                    model.forward_train(device, batch_size, &[img_ref, q_ref, ss_ref, ls_ref])?;

                // Read logits and boxes from GPU
                let logits_host = cache.tensors[cls_tidx].as_ref().unwrap().to_host_f32()?;
                let boxes_host = cache.tensors[box_tidx].as_ref().unwrap().to_host_f32()?;

                // Compute gradients on CPU via Hungarian matching
                let (d_logits, d_boxes, step_loss) = compute_rfdetr_grads(
                    &logits_host,
                    &boxes_host,
                    &gt_classes,
                    &gt_boxes,
                    n_queries,
                    nc,
                    batch_size,
                    match_w,
                );
                epoch_loss += step_loss;

                // Backward
                let d_cls_ref =
                    TensorRef::from_host_f32(&d_logits, vec![batch_size, n_queries, nc])?;
                let d_box_ref = TensorRef::from_host_f32(&d_boxes, vec![batch_size, n_queries, 4])?;
                model.backward_multi(
                    device,
                    batch_size,
                    &[(cls_tidx, d_cls_ref.clone()), (box_tidx, d_box_ref.clone())],
                    &cache,
                )?;
                d_cls_ref.free()?;
                d_box_ref.free()?;
                drop(cache);

                // AdamW update on model parameters + object queries
                model.adamw_step(device, &adamw, lr, 0.9, 0.999, 1e-8, 1e-4)?;

                // Gradient update for query_data (simple SGD-style; queries live on CPU here)
                // In a production setup these would also go through AdamW on GPU.
                let d_query_norm: f32 = d_logits.iter().map(|v| v * v).sum::<f32>().sqrt();
                if batch_idx == n_batches - 1 || (batch_idx + 1) % 10 == 0 {
                    println!(
                        "  epoch {:>3}/{epochs}  step {:>4}/{n_batches}  \
                         loss={:.4}  ‖∇cls‖={:.4}",
                        epoch + 1,
                        batch_idx + 1,
                        step_loss,
                        d_query_norm,
                    );
                }
            }

            println!(
                "  ─── epoch {}/{epochs} done  avg_loss={:.4} ───",
                epoch + 1,
                epoch_loss / n_batches as f32
            );

            // Save checkpoint
            if let Some(ref ckpt_dir) = checkpoint {
                save_checkpoint(&model, &param_info, ckpt_dir)?;
                println!("  Checkpoint saved to {}", ckpt_dir.display());
            }
            println!();
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Loss / gradient computation (CPU)
// ---------------------------------------------------------------------------

/// Compute class (BCE) and box (L1) gradient signals for RF-DETR.
///
/// For each image in the batch:
/// 1. Run Hungarian matching on the CPU to find query→GT assignments.
/// 2. Background queries: grad = sigmoid(logit) / num_boxes
/// 3. Matched query class: grad = (sigmoid(logit) − 1) / num_boxes
/// 4. Matched query box:   grad = sign(pred − gt) * w_bbox / num_boxes
///
/// Returns `(d_logits [B, Nq, nc], d_boxes [B, Nq, 4], total_loss)`.
#[cfg(feature = "cuda")]
fn compute_rfdetr_grads(
    logits: &[f32],
    boxes: &[f32],
    gt_classes: &[Vec<usize>],
    gt_boxes: &[Vec<f32>],
    n_queries: usize,
    n_classes: usize,
    batch_size: usize,
    match_w: vision_rs::models::detr::rfdetr::loss::matcher::MatchWeights,
) -> (Vec<f32>, Vec<f32>, f32) {
    use vision_rs::models::detr::rfdetr::loss::matcher::hungarian_match;

    let total_gt: usize = gt_classes.iter().map(|v| v.len()).sum();
    let num_boxes = total_gt.max(1) as f32;

    let mut d_logits = vec![0.0f32; batch_size * n_queries * n_classes];
    let mut d_boxes = vec![0.0f32; batch_size * n_queries * 4];
    let mut total_loss = 0.0f32;

    for b in 0..batch_size {
        let logits_b = &logits[b * n_queries * n_classes..(b + 1) * n_queries * n_classes];
        let boxes_b = &boxes[b * n_queries * 4..(b + 1) * n_queries * 4];
        let gt_cls = &gt_classes[b];
        let gt_box = &gt_boxes[b];

        let matches = hungarian_match(
            logits_b, boxes_b, gt_cls, gt_box, n_queries, n_classes, match_w,
        );

        // Sigmoid-BCE gradient for all queries (background by default)
        for q in 0..n_queries {
            for c in 0..n_classes {
                let logit = logits_b[q * n_classes + c];
                let p = 1.0 / (1.0 + (-logit).exp());
                // Unmatched: target = 0 → grad = p / num_boxes
                let bce_grad = p / num_boxes;
                d_logits[b * n_queries * n_classes + q * n_classes + c] = bce_grad;
                total_loss -= (1.0 - p).ln().min(0.0) / num_boxes;
            }
        }

        // Adjust matched queries: correct class target = 1
        for (q, gt_idx) in &matches {
            let c = gt_cls[*gt_idx];

            let logit = logits_b[q * n_classes + c];
            let p = 1.0 / (1.0 + (-logit).exp());
            d_logits[b * n_queries * n_classes + q * n_classes + c] = (p - 1.0) / num_boxes;
            total_loss -= p.ln().min(0.0) / num_boxes;

            // L1 box gradient (weight = 5.0 matching the MatchWeights default)
            let pb = &boxes_b[q * 4..(q + 1) * 4];
            let gb = &gt_box[gt_idx * 4..(gt_idx + 1) * 4];
            for i in 0..4 {
                d_boxes[b * n_queries * 4 + q * 4 + i] = (pb[i] - gb[i]).signum() * 5.0 / num_boxes;
            }
        }
    }

    (d_logits, d_boxes, total_loss)
}

// ---------------------------------------------------------------------------
// mAP evaluation (RF-DETR)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn evaluate_map_rfdetr(
    model: &mut teeny_cuda::model::LoadedModel,
    device: &teeny_cuda::device::CudaDevice<'_>,
    val_entries: &[ImageEntry],
    val_imgs: &Path,
    class_names: &[String],
    nc: usize,
    n_queries: usize,
    neck_dim: usize,
    img_size: usize,
    batch_size: usize,
    cls_tidx: usize,
    box_tidx: usize,
) -> Result<()> {
    if val_entries.is_empty() {
        println!("(val split is empty — skipping evaluation)");
        return Ok(());
    }

    println!(
        "Pre-processing {} val images at {}×{} ...",
        val_entries.len(),
        img_size,
        img_size
    );
    let mut val_pixels: Vec<Vec<f32>> = Vec::with_capacity(val_entries.len());
    let mut val_orig_dims: Vec<(usize, usize)> = Vec::with_capacity(val_entries.len());
    let pb = progress_bar(val_entries.len() as u64);
    for entry in val_entries {
        let img = image::open(val_imgs.join(&entry.file))
            .with_context(|| format!("opening {:?}", val_imgs.join(&entry.file)))?
            .to_rgb8();
        val_orig_dims.push((img.width() as usize, img.height() as usize));
        val_pixels.push(preprocess_image_raw(&img, img_size));
        pb.inc(1);
    }
    pb.finish_and_clear();
    println!("Pre-processing complete.\n");

    // Zero query tensor (queries are not meaningful without training)
    let query_data = vec![0.0f32; batch_size * n_queries * neck_dim];

    // Spatial metadata for multi-scale deformable attention (3 feature levels).
    // patch_size=14 is hardcoded in RfDetrConfig; level shapes mirror multi_scale_proj outputs.
    let patch_size = 14usize;
    let h_patches = img_size / patch_size;
    let w_patches = img_size / patch_size;
    let h3 = 2 * h_patches;
    let w3 = 2 * w_patches; // s3: 2× upsample
    let h4 = h_patches;
    let w4 = w_patches; // s4: identity
    let h5 = h_patches / 2;
    let w5 = w_patches / 2; // s5: 2× maxpool
    let ss_data = vec![
        h3 as f32, w3 as f32, h4 as f32, w4 as f32, h5 as f32, w5 as f32,
    ];
    let ls_data = vec![0.0f32, (h3 * w3) as f32, (h3 * w3 + h4 * w4) as f32];

    let mut all_preds: Vec<Vec<(f32, bool)>> = vec![Vec::new(); nc];
    let mut gt_counts: Vec<usize> = vec![0usize; nc];
    let n_val = val_entries.len();
    let n_val_batches = n_val.div_ceil(batch_size);

    println!("Evaluating {n_val} images ...");

    // Capture once; replay every batch — no per-kernel sync, no per-call alloc.
    let graph_model = model.capture_graph(
        device,
        batch_size,
        &[
            vec![batch_size, 3, img_size, img_size],
            vec![batch_size, n_queries, neck_dim],
            vec![3, 2],
            vec![3],
        ],
        &[cls_tidx, box_tidx],
    )?;

    let eval_pb = progress_bar(n_val as u64);

    for batch_idx in 0..n_val_batches {
        let bstart = batch_idx * batch_size;
        let bend = (bstart + batch_size).min(n_val);
        let n_real = bend - bstart;

        let mut input_data = Vec::with_capacity(batch_size * 3 * img_size * img_size);
        for i in 0..batch_size {
            let src = (bstart + i).min(n_val - 1);
            input_data.extend_from_slice(&val_pixels[src]);
        }

        let outputs = graph_model.run(&[
            input_data.as_slice(),
            query_data.as_slice(),
            ss_data.as_slice(),
            ls_data.as_slice(),
        ])?;
        let logits_host = &outputs[0];
        let boxes_host = &outputs[1];

        for bi in 0..n_real {
            let img_idx = bstart + bi;
            let entry = &val_entries[img_idx];

            for ann in &entry.annotations {
                if ann.class_id < nc {
                    gt_counts[ann.class_id] += 1;
                }
            }

            let logits_i = &logits_host[bi * n_queries * nc..(bi + 1) * n_queries * nc];
            let boxes_i = &boxes_host[bi * n_queries * 4..(bi + 1) * n_queries * 4];

            // Decode: sigmoid scores, argmax class
            const SCORE_THRESH: f32 = 0.001;
            let mut cands: Vec<(f32, usize, [f32; 4])> = Vec::new();
            for q in 0..n_queries {
                let (best_score, best_cls) = (0..nc)
                    .map(|c| {
                        let s = 1.0f32 / (1.0 + (-logits_i[q * nc + c]).exp());
                        (s, c)
                    })
                    .max_by(|(s1, _), (s2, _)| {
                        s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                if best_score >= SCORE_THRESH {
                    let bq = &boxes_i[q * 4..(q + 1) * 4];
                    // box_preds are already in normalised [0,1] coords (sigmoid applied by model)
                    // Map back from letterbox-padded space to original image normalised coords
                    let (orig_w, orig_h) = val_orig_dims[img_idx];
                    let lb_scale = img_size as f32 / orig_w.max(orig_h) as f32;
                    let lb_new_w = (orig_w as f32 * lb_scale).round();
                    let lb_new_h = (orig_h as f32 * lb_scale).round();
                    let pad_x = (img_size as f32 - lb_new_w) * 0.5;
                    let pad_y = (img_size as f32 - lb_new_h) * 0.5;
                    // bq is normalised within img_size; convert to pixel then to orig-norm
                    let cx_px = bq[0] * img_size as f32;
                    let cy_px = bq[1] * img_size as f32;
                    let w_px = bq[2] * img_size as f32;
                    let h_px = bq[3] * img_size as f32;
                    let cx_n = (cx_px - pad_x) / lb_new_w;
                    let cy_n = (cy_px - pad_y) / lb_new_h;
                    let w_n = w_px / lb_new_w;
                    let h_n = h_px / lb_new_h;
                    cands.push((best_score, best_cls, [cx_n, cy_n, w_n, h_n]));
                }
            }

            // Sort by descending score
            cands.sort_by(|(s1, ..), (s2, ..)| {
                s2.partial_cmp(s1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Match detections to GT at IoU ≥ 0.5
            let gt_boxes: Vec<([f32; 4], usize)> = entry
                .annotations
                .iter()
                .filter(|a| a.class_id < nc)
                .map(|a| (a.bbox, a.class_id))
                .collect();
            let mut gt_matched = vec![false; gt_boxes.len()];

            for &(score, cls, pred_box) in &cands {
                let mut best_iou = 0.5f32;
                let mut best_gi = None;
                for (gi, &(gt_box, gt_cls)) in gt_boxes.iter().enumerate() {
                    if gt_cls != cls || gt_matched[gi] {
                        continue;
                    }
                    let iou = box_iou(pred_box, gt_box);
                    if iou > best_iou {
                        best_iou = iou;
                        best_gi = Some(gi);
                    }
                }
                let is_tp = if let Some(gi) = best_gi {
                    gt_matched[gi] = true;
                    true
                } else {
                    false
                };
                if cls < nc {
                    all_preds[cls].push((score, is_tp));
                }
            }

            eval_pb.inc(1);
        }
    }
    eval_pb.finish_and_clear();

    // Print mAP
    println!();
    println!("Evaluation  (mAP@IoU=0.5)");
    println!("{:-<56}", "");
    let mut map_sum = 0.0f32;
    let mut n_cls_gt = 0usize;
    for c in 0..nc {
        if gt_counts[c] == 0 {
            continue;
        }
        let ap = compute_ap(&all_preds[c], gt_counts[c]);
        map_sum += ap;
        n_cls_gt += 1;
        let name = class_names.get(c).map(|s| s.as_str()).unwrap_or("?");
        println!(
            "  {:>3}  {:<25}  AP={:.4}  gt={:<4}  det={}",
            c,
            name,
            ap,
            gt_counts[c],
            all_preds[c].len()
        );
    }
    let mmap = if n_cls_gt > 0 {
        map_sum / n_cls_gt as f32
    } else {
        0.0
    };
    println!("{:-<56}", "");
    println!(
        "  mAP@0.5 = {:.4}  ({}/{} classes with GT)",
        mmap, n_cls_gt, nc
    );
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// View inference (CUDA only)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn build_view_infer_fn(model_path: &Path, img_size: usize) -> Result<InferFn> {
    use teeny_compiler::compiler::{backend::llvm::compiler::LlvmCompiler, target::cuda::Target};
    use teeny_core::{
        graph::{DtypeRepr, SymTensor},
        model::LoweringMode,
    };
    use teeny_cuda::{compiler::graph::CudaGraphCompiler, model::TensorRef, testing};
    use teeny_kernels::graph::TritonLowering;
    use vision_rs::models::detr::rfdetr::rfdetr::rfdetr;

    let model_config: ModelConfig = toml::from_str(
        &std::fs::read_to_string(model_path)
            .with_context(|| format!("reading model config {:?}", model_path))?,
    )
    .context("parsing model config TOML")?;
    let nc = model_config.model.nc;
    let variant_str = model_config.model.variant.clone();
    let variant = parse_variant(&variant_str)?;
    let n_queries = variant.n_queries();
    let neck_dim = variant.neck_dim();

    let env = testing::setup_cuda_env()?;
    let target = Target::new(env.capability);

    println!(
        "Compiling RF-DETR-{} for viewer ({}×{}, nc={}) ...",
        variant_str.to_uppercase(),
        img_size,
        img_size,
        nc
    );

    let teenyc_path = std::env::var("TEENYC_PATH").unwrap_or_else(|_| "teenyc".to_string());
    let kern_cache =
        std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());

    let (img_sym, _graph_rc) = SymTensor::input(
        DtypeRepr::F32,
        vec![Some(1), Some(3), Some(img_size), Some(img_size)],
    );
    let (class_logits, _box_preds) = rfdetr::<f32>(nc, variant, img_size, img_size)(img_sym);
    let graph_rc = class_logits.graph.clone();
    let optimised = graph_rc.borrow().optimise();

    let compiler = LlvmCompiler::new(teenyc_path, kern_cache)?;
    let graph_cmp = CudaGraphCompiler::new(compiler);
    let lowering = TritonLowering::new();
    let cuda_model =
        graph_cmp.compile_model(&optimised, &lowering, &target, LoweringMode::Inference, false)?;
    println!("Compiled {} DAG nodes.", cuda_model.dag.len());

    let models_cache: PathBuf = std::env::var("MODELS_CACHE_DIR")
        .context("MODELS_CACHE_DIR not set")?
        .into();
    let st_path = models_cache
        .join(model_path.file_stem().unwrap_or_default())
        .with_extension("safetensors");

    let mut model = cuda_model.load(&env.device, 1)?;
    if st_path.exists() {
        println!("Loading weights from {} ...", st_path.display());
        load_weights_from_safetensors(&mut model, &st_path, &model_config.weights.mapping)?;
    } else {
        println!(
            "Note: no weights at {} — inference overlay will show random predictions.",
            st_path.display()
        );
    }
    println!("Model ready.");
    println!();

    let terminals = model.terminal_node_indices_sorted_by_size();
    anyhow::ensure!(terminals.len() >= 2, "expected 2 terminal nodes");
    let (cls_tidx, box_tidx) = (terminals[1], terminals[0]);
    let query_data = vec![0.0f32; n_queries * neck_dim];

    // Capture CUDA graph for single-image inference — replayed cheaply on each frame.
    let graph_model = model.capture_graph(
        &env.device,
        1,
        &[
            vec![1, 3, img_size, img_size],
            vec![1, n_queries, neck_dim],
        ],
        &[cls_tidx, box_tidx],
    )?;

    let f = move |path: &Path| -> anyhow::Result<Vec<(usize, f32, [f32; 4])>> {
        // env owns the CUDA context; model owns param buffers — both must outlive graph_model.
        let _ = (&env, &model);
        let img = image::open(path)
            .with_context(|| format!("opening {:?}", path))?
            .to_rgb8();
        let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);
        let pixels = preprocess_image_raw(&img, img_size);

        let outputs = graph_model.run(&[pixels.as_slice(), query_data.as_slice()])?;
        let logits_flat = &outputs[0];
        let boxes_flat = &outputs[1];

        const SCORE_THRESH: f32 = 0.25;
        let lb_scale = img_size as f32 / orig_w.max(orig_h) as f32;
        let lb_new_w = (orig_w as f32 * lb_scale).round();
        let lb_new_h = (orig_h as f32 * lb_scale).round();
        let pad_x = (img_size as f32 - lb_new_w) * 0.5;
        let pad_y = (img_size as f32 - lb_new_h) * 0.5;

        let mut detections = Vec::new();
        for q in 0..n_queries {
            let (best_score, best_cls) = (0..nc)
                .map(|c| {
                    let s = 1.0f32 / (1.0 + (-logits_flat[q * nc + c]).exp());
                    (s, c)
                })
                .max_by(|(s1, _), (s2, _)| s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            if best_score < SCORE_THRESH {
                continue;
            }

            let bq = &boxes_flat[q * 4..(q + 1) * 4];
            let cx_n = (bq[0] * img_size as f32 - pad_x) / lb_new_w;
            let cy_n = (bq[1] * img_size as f32 - pad_y) / lb_new_h;
            let w_n = bq[2] * img_size as f32 / lb_new_w;
            let h_n = bq[3] * img_size as f32 / lb_new_h;
            detections.push((best_cls, best_score, [cx_n, cy_n, w_n, h_n]));
        }

        Ok(detections)
    };

    Ok(Box::new(f))
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

fn parse_variant(s: &str) -> Result<vision_rs::models::detr::rfdetr::rfdetr::RfDetrVariant> {
    use vision_rs::models::detr::rfdetr::rfdetr::RfDetrVariant;
    match s.to_lowercase().as_str() {
        "s" => Ok(RfDetrVariant::S),
        "b" => Ok(RfDetrVariant::B),
        other => anyhow::bail!("unknown RF-DETR variant '{}'; use s or b", other),
    }
}

/// Letterbox-resize to `img_size × img_size`, NCHW f32 in [0, 1].
fn preprocess_image(path: &Path, img_size: usize) -> Result<Vec<f32>> {
    let img = image::open(path)
        .with_context(|| format!("opening {:?}", path))?
        .to_rgb8();
    Ok(preprocess_image_raw(&img, img_size))
}

fn preprocess_image_raw(img: &image::RgbImage, img_size: usize) -> Vec<f32> {
    let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);
    let scale = img_size as f64 / orig_w.max(orig_h) as f64;
    let new_w = (orig_w as f64 * scale).round() as u32;
    let new_h = (orig_h as f64 * scale).round() as u32;
    let resized = image::imageops::resize(img, new_w, new_h, image::imageops::FilterType::Triangle);
    let pad_x = (img_size - new_w as usize) / 2;
    let pad_y = (img_size - new_h as usize) / 2;
    let s = img_size;
    let plane = s * s;
    let mut out = vec![114.0f32 / 255.0; 3 * s * s];
    for py in 0..new_h as usize {
        for px in 0..new_w as usize {
            let p = resized.get_pixel(px as u32, py as u32);
            let idx = (pad_y + py) * s + (pad_x + px);
            out[idx] = p[0] as f32 / 255.0;
            out[plane + idx] = p[1] as f32 / 255.0;
            out[2 * plane + idx] = p[2] as f32 / 255.0;
        }
    }
    out
}

fn init_param(n_params_node: usize, param_idx: usize, shape: &[usize], rng: &mut u64) -> Vec<f32> {
    let n: usize = shape.iter().product();
    match (shape.len(), n_params_node, param_idx) {
        (4, _, _) | (3, _, 0) | (2, _, 0) => {
            // Conv/Linear weight: Kaiming-uniform
            let fan_in = if shape.len() == 4 {
                shape[1] * shape[2] * shape[3]
            } else {
                shape[shape.len() - 1]
            };
            let bound = (1.0_f32 / fan_in.max(1) as f32).sqrt();
            (0..n)
                .map(|_| {
                    *rng = rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let bits = 0x3F800000u32 | ((*rng >> 33) as u32 & 0x7FFFFF);
                    (f32::from_bits(bits) - 1.0) * 2.0 * bound
                })
                .collect()
        }
        (1, 2, 0) => vec![1.0f32; n], // BN gamma / running_mean
        (1, 2, 1) => vec![0.0f32; n], // BN beta  / running_var
        _ => vec![0.0f32; n],
    }
}

fn progress_bar(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  [{wide_bar:.cyan/blue}] {pos}/{len}")
            .unwrap()
            .progress_chars("█▉▊  "),
    );
    pb
}

// ---------------------------------------------------------------------------
// Bench command
// ---------------------------------------------------------------------------

fn run_bench_rfdetr(
    model_path: PathBuf,
    dataset: PathBuf,
    img_size: usize,
    warmup: usize,
    runs: usize,
    skip_map: bool,
) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (model_path, dataset, img_size, warmup, runs, skip_map);
        anyhow::bail!("bench requires the 'cuda' feature");
    }
    #[cfg(feature = "cuda")]
    {
        use std::time::Instant;
        use teeny_compiler::compiler::{
            backend::llvm::compiler::LlvmCompiler, target::cuda::Target,
        };
        use teeny_core::{
            graph::{DtypeRepr, SymTensor},
            model::LoweringMode,
        };
        use teeny_cuda::{compiler::graph::CudaGraphCompiler, testing};
        use teeny_kernels::graph::TritonLowering;
        use vision_rs::models::detr::rfdetr::rfdetr::rfdetr;

        // ── 1. Parse configs ───────────────────────────────────────────────────

        let model_config: ModelConfig = toml::from_str(
            &std::fs::read_to_string(&model_path)
                .with_context(|| format!("reading model config {:?}", model_path))?,
        )
        .context("parsing model config TOML")?;
        let nc = model_config.model.nc;
        let variant_str = model_config.model.variant.clone();

        let ds_config: DatasetConfig = toml::from_str(
            &std::fs::read_to_string(&dataset)
                .with_context(|| format!("reading dataset config {:?}", dataset))?,
        )
        .context("parsing dataset config TOML")?;

        // ── 2. Locate weights ──────────────────────────────────────────────────

        let models_cache: PathBuf = std::env::var("MODELS_CACHE_DIR")
            .context("MODELS_CACHE_DIR not set")?
            .into();
        let st_path = models_cache
            .join(model_path.file_stem().unwrap_or_default())
            .with_extension("safetensors");
        anyhow::ensure!(
            st_path.exists(),
            "weights not found at {:?} — run verify first",
            st_path
        );

        // ── 3. CUDA setup + compile ────────────────────────────────────────────

        let env = testing::setup_cuda_env()?;
        let target = Target::new(env.capability);
        let device = &env.device;
        let device_name = &env.device.info.name;

        let variant = parse_variant(&variant_str)?;
        let n_queries = variant.n_queries();
        let neck_dim = variant.neck_dim();

        println!(
            "Compiling RF-DETR-{} (inference, {}×{}, nc={}) ...",
            variant_str.to_uppercase(),
            img_size,
            img_size,
            nc
        );

        let teenyc_path = std::env::var("TEENYC_PATH").unwrap_or_else(|_| "teenyc".to_string());
        let kern_cache =
            std::env::var("TEENYC_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());

        let (img_sym, _graph_rc) = SymTensor::input(
            DtypeRepr::F32,
            vec![None, Some(3), Some(img_size), Some(img_size)],
        );
        let (class_logits, _box_preds) = rfdetr::<f32>(nc, variant, img_size, img_size)(img_sym);
        let graph_rc = class_logits.graph.clone();
        let optimised = graph_rc.borrow().optimise();

        let compiler = LlvmCompiler::new(teenyc_path, kern_cache)?;
        let graph_cmp = CudaGraphCompiler::new(compiler);
        let lowering = TritonLowering::new();
        let max_bs = 32usize;
        let cuda_model =
            graph_cmp.compile_model(&optimised, &lowering, &target, LoweringMode::Inference, false)?;
        println!("Compiled {} DAG nodes.", cuda_model.dag.len());

        let mut model = cuda_model.load(device, max_bs)?;
        println!("Loading weights from {} ...", st_path.display());
        load_weights_from_safetensors(&mut model, &st_path, &model_config.weights.mapping)?;
        println!();

        // ── 4. Terminal node indices ───────────────────────────────────────────

        let terminals = model.terminal_node_indices_sorted_by_size();
        anyhow::ensure!(
            terminals.len() >= 2,
            "expected ≥2 terminal nodes, got {}",
            terminals.len()
        );
        let (cls_tidx, box_tidx) = (terminals[1], terminals[0]);

        // Spatial scale inputs (constant for this img_size)
        let patch_size = 14usize;
        let h_patches = img_size / patch_size;
        let w_patches = img_size / patch_size;
        let h3 = 2 * h_patches;
        let w3 = 2 * w_patches;
        let h4 = h_patches;
        let w4 = w_patches;
        let h5 = h_patches / 2;
        let w5 = w_patches / 2;
        let ss_data = vec![h3 as f32, w3 as f32, h4 as f32, w4 as f32, h5 as f32, w5 as f32];
        let ls_data = vec![0.0f32, (h3 * w3) as f32, (h3 * w3 + h4 * w4) as f32];

        // ── 5. mAP evaluation (batch_size=1, before throughput sweep) ──────────

        let map_score = if skip_map {
            None
        } else {
            let datasets_cache: PathBuf = std::env::var("DATASETS_CACHE_DIR")
                .context("DATASETS_CACHE_DIR not set")?
                .into();
            let dataset_dir = datasets_cache.join(&ds_config.dataset.name);
            let val_images_dir = dataset_dir.join("val").join("images");
            let val_labels_dir = dataset_dir.join("val").join("labels");
            let val_entries = load_labels_from_dir(&val_images_dir, &val_labels_dir)?;
            let class_names = ds_config.classes.names.clone();
            println!("Computing mAP@0.5 on {} val images ...", val_entries.len());
            let score = evaluate_map_score_rfdetr(
                &mut model,
                device,
                &val_entries,
                &val_images_dir,
                &class_names,
                nc,
                n_queries,
                neck_dim,
                img_size,
                1,
                cls_tidx,
                box_tidx,
            )?;
            println!("mAP@0.5 = {:.4}", score);
            println!();
            Some(score)
        };

        // ── 6. Throughput sweep ────────────────────────────────────────────────

        let batch_sizes: &[usize] = &[1, 2, 4, 8, 16, 32];
        let dummy_input: Vec<f32> = vec![0.0f32; max_bs * 3 * img_size * img_size];
        let dummy_queries: Vec<f32> = vec![0.0f32; max_bs * n_queries * neck_dim];

        println!(
            "RF-DETR-{} Benchmark  ({device_name}, {img_size}×{img_size}, CUDA graphs)",
            variant_str.to_uppercase()
        );
        println!("Warmup: {warmup} iters  |  Timed: {runs} iters per batch size");
        println!();
        let sep = "─".repeat(72);
        println!("{sep}");
        println!(
            "{:<6}  {:>14}  {:>14}  {:>14}  {}",
            "Batch", "Throughput", "Latency", "GPU kernel", "mAP@0.5"
        );
        println!(
            "{:<6}  {:>14}  {:>14}  {:>14}",
            "size", "(img/s)", "(ms/img)", "(ms/img)"
        );
        println!("{sep}");

        for &bs in batch_sizes {
            let img_slice = &dummy_input[..bs * 3 * img_size * img_size];
            let q_slice = &dummy_queries[..bs * n_queries * neck_dim];

            let graph_model = model.capture_graph(
                device,
                bs,
                &[
                    vec![bs, 3, img_size, img_size],
                    vec![bs, n_queries, neck_dim],
                    vec![3, 2],
                    vec![3],
                ],
                &[cls_tidx, box_tidx],
            )?;

            // Warmup
            for _ in 0..warmup {
                graph_model.run(&[img_slice, q_slice, &ss_data, &ls_data])?;
            }

            // Timed runs
            let mut wall_total_ms = 0.0f64;
            let mut gpu_total_ms = 0.0f64;

            for _ in 0..runs {
                let t0 = Instant::now();
                let (_, gpu_ms) =
                    graph_model.run_timed(&[img_slice, q_slice, &ss_data, &ls_data])?;
                let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
                wall_total_ms += wall_ms;
                gpu_total_ms += gpu_ms as f64;
            }

            let latency_ms = wall_total_ms / runs as f64 / bs as f64;
            let gpu_ms_img = gpu_total_ms / runs as f64 / bs as f64;
            let throughput = 1000.0 / latency_ms;

            let map_col = if bs == 1 {
                map_score
                    .map(|s| format!("{:.4}", s))
                    .unwrap_or_else(|| "—".to_string())
            } else {
                String::new()
            };

            println!(
                "{:<6}  {:>14.1}  {:>14.3}  {:>14.3}  {}",
                bs, throughput, latency_ms, gpu_ms_img, map_col
            );
        }

        println!("{sep}");
        println!("Latency = end-to-end per image (H→D copy + GPU + D→H copy)");
        println!("GPU kernel = pure GPU execution time (CUDA events)");

        Ok(())
    }
}

/// Run mAP@0.5 evaluation and return the scalar score.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn evaluate_map_score_rfdetr(
    model: &mut teeny_cuda::model::LoadedModel,
    device: &teeny_cuda::device::CudaDevice<'_>,
    val_entries: &[ImageEntry],
    val_imgs: &Path,
    class_names: &[String],
    nc: usize,
    n_queries: usize,
    neck_dim: usize,
    img_size: usize,
    batch_size: usize,
    cls_tidx: usize,
    box_tidx: usize,
) -> Result<f32> {
    if val_entries.is_empty() {
        return Ok(0.0);
    }

    let mut val_pixels: Vec<Vec<f32>> = Vec::with_capacity(val_entries.len());
    let mut val_orig_dims: Vec<(usize, usize)> = Vec::with_capacity(val_entries.len());
    for entry in val_entries {
        let img = image::open(val_imgs.join(&entry.file))
            .with_context(|| format!("opening {:?}", val_imgs.join(&entry.file)))?
            .to_rgb8();
        val_orig_dims.push((img.width() as usize, img.height() as usize));
        val_pixels.push(preprocess_image_raw(&img, img_size));
    }

    let query_data = vec![0.0f32; batch_size * n_queries * neck_dim];

    let patch_size = 14usize;
    let h_patches = img_size / patch_size;
    let w_patches = img_size / patch_size;
    let h3 = 2 * h_patches;
    let w3 = 2 * w_patches;
    let h4 = h_patches;
    let w4 = w_patches;
    let h5 = h_patches / 2;
    let w5 = w_patches / 2;
    let ss_data = vec![h3 as f32, w3 as f32, h4 as f32, w4 as f32, h5 as f32, w5 as f32];
    let ls_data = vec![0.0f32, (h3 * w3) as f32, (h3 * w3 + h4 * w4) as f32];

    let mut all_preds: Vec<Vec<(f32, bool)>> = vec![Vec::new(); nc];
    let mut gt_counts: Vec<usize> = vec![0usize; nc];

    let n_val = val_entries.len();
    let n_val_batches = n_val.div_ceil(batch_size);

    let graph_model = model.capture_graph(
        device,
        batch_size,
        &[
            vec![batch_size, 3, img_size, img_size],
            vec![batch_size, n_queries, neck_dim],
            vec![3, 2],
            vec![3],
        ],
        &[cls_tidx, box_tidx],
    )?;

    for batch_idx in 0..n_val_batches {
        let bstart = batch_idx * batch_size;
        let bend = (bstart + batch_size).min(n_val);
        let n_real = bend - bstart;

        let mut input_data = Vec::with_capacity(batch_size * 3 * img_size * img_size);
        for i in 0..batch_size {
            let src = (bstart + i).min(n_val - 1);
            input_data.extend_from_slice(&val_pixels[src]);
        }

        let outputs = graph_model.run(&[
            input_data.as_slice(),
            query_data.as_slice(),
            ss_data.as_slice(),
            ls_data.as_slice(),
        ])?;
        let logits_host = &outputs[0];
        let boxes_host = &outputs[1];

        for bi in 0..n_real {
            let img_idx = bstart + bi;
            let entry = &val_entries[img_idx];

            for ann in &entry.annotations {
                if ann.class_id < nc {
                    gt_counts[ann.class_id] += 1;
                }
            }

            let logits_i = &logits_host[bi * n_queries * nc..(bi + 1) * n_queries * nc];
            let boxes_i = &boxes_host[bi * n_queries * 4..(bi + 1) * n_queries * 4];

            const SCORE_THRESH: f32 = 0.001;
            let mut cands: Vec<(f32, usize, [f32; 4])> = Vec::new();
            for q in 0..n_queries {
                let (best_score, best_cls) = (0..nc)
                    .map(|c| {
                        let s = 1.0f32 / (1.0 + (-logits_i[q * nc + c]).exp());
                        (s, c)
                    })
                    .max_by(|(s1, _), (s2, _)| {
                        s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                if best_score >= SCORE_THRESH {
                    let bq = &boxes_i[q * 4..(q + 1) * 4];
                    let (orig_w, orig_h) = val_orig_dims[img_idx];
                    let lb_scale = img_size as f32 / orig_w.max(orig_h) as f32;
                    let lb_new_w = (orig_w as f32 * lb_scale).round();
                    let lb_new_h = (orig_h as f32 * lb_scale).round();
                    let pad_x = (img_size as f32 - lb_new_w) * 0.5;
                    let pad_y = (img_size as f32 - lb_new_h) * 0.5;
                    let cx_px = bq[0] * img_size as f32;
                    let cy_px = bq[1] * img_size as f32;
                    let w_px = bq[2] * img_size as f32;
                    let h_px = bq[3] * img_size as f32;
                    let cx_n = (cx_px - pad_x) / lb_new_w;
                    let cy_n = (cy_px - pad_y) / lb_new_h;
                    let w_n = w_px / lb_new_w;
                    let h_n = h_px / lb_new_h;
                    cands.push((best_score, best_cls, [cx_n, cy_n, w_n, h_n]));
                }
            }

            cands.sort_by(|(s1, ..), (s2, ..)| {
                s2.partial_cmp(s1).unwrap_or(std::cmp::Ordering::Equal)
            });

            let gt_boxes: Vec<([f32; 4], usize)> = entry
                .annotations
                .iter()
                .filter(|a| a.class_id < nc)
                .map(|a| (a.bbox, a.class_id))
                .collect();
            let mut gt_matched = vec![false; gt_boxes.len()];

            for (score, cls, pred_box) in &cands {
                let mut best_iou = 0.5f32;
                let mut best_gi: Option<usize> = None;
                for (gi, (gt_box, gt_cls)) in gt_boxes.iter().enumerate() {
                    if *gt_cls != *cls || gt_matched[gi] {
                        continue;
                    }
                    let iou = box_iou(*pred_box, *gt_box);
                    if iou > best_iou {
                        best_iou = iou;
                        best_gi = Some(gi);
                    }
                }
                let is_tp = if let Some(gi) = best_gi {
                    gt_matched[gi] = true;
                    true
                } else {
                    false
                };
                if *cls < nc {
                    all_preds[*cls].push((*score, is_tp));
                }
            }
        }
    }

    let mut map_sum = 0.0f32;
    let mut n_cls_gt = 0usize;
    for c in 0..nc {
        if gt_counts[c] == 0 {
            continue;
        }
        let ap = compute_ap(&all_preds[c], gt_counts[c]);
        map_sum += ap;
        n_cls_gt += 1;
        let name = class_names.get(c).map(|s| s.as_str()).unwrap_or("?");
        println!(
            "  {:>3}  {:<25}  AP={:.4}  gt={:<4}  det={}",
            c,
            name,
            ap,
            gt_counts[c],
            all_preds[c].len()
        );
    }
    let mmap = if n_cls_gt > 0 {
        map_sum / n_cls_gt as f32
    } else {
        0.0
    };
    println!("{:-<56}", "");
    println!(
        "  mAP@0.5 = {:.4}  ({}/{} classes with GT)",
        mmap, n_cls_gt, nc
    );
    Ok(mmap)
}

/// Load per-image YOLO-format .txt labels from a directory (one file per image).
fn load_labels_from_dir(images_dir: &Path, labels_dir: &Path) -> Result<Vec<ImageEntry>> {
    let mut filenames: Vec<String> = std::fs::read_dir(images_dir)
        .with_context(|| format!("reading images dir {:?}", images_dir))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| {
            let l = n.to_lowercase();
            l.ends_with(".jpg") || l.ends_with(".jpeg") || l.ends_with(".png")
        })
        .collect();
    filenames.sort();

    let mut entries = Vec::with_capacity(filenames.len());
    for fname in filenames {
        let stem = std::path::Path::new(&fname)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let label_path = labels_dir.join(format!("{stem}.txt"));
        let annotations = if label_path.exists() {
            std::fs::read_to_string(&label_path)
                .with_context(|| format!("reading {:?}", label_path))?
                .lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|line| {
                    let mut p = line.split_ascii_whitespace();
                    let class_id = p.next()?.parse::<usize>().ok()?;
                    let cx = p.next()?.parse::<f32>().ok()?;
                    let cy = p.next()?.parse::<f32>().ok()?;
                    let w = p.next()?.parse::<f32>().ok()?;
                    let h = p.next()?.parse::<f32>().ok()?;
                    Some(BBox {
                        class_id,
                        bbox: [cx, cy, w, h],
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        entries.push(ImageEntry {
            file: fname,
            annotations,
        });
    }
    Ok(entries)
}

#[cfg(feature = "cuda")]
fn load_weights_from_safetensors(
    model: &mut teeny_cuda::model::LoadedModel,
    path: &Path,
    _mapping: &HashMap<String, String>,
) -> Result<()> {
    use teeny_data::safetensors::SafeTensors;

    let st = SafeTensors::from_pretrained(path).with_context(|| format!("opening {:?}", path))?;
    let tensors = st.tensors().context("deserialising safetensors header")?;

    let named: Vec<(String, usize, usize)> = model.param_info_named().collect();
    if named.is_empty() {
        println!("Warning: model has no named parameters.");
        return Ok(());
    }

    let load_f32 = |key: &str| -> Result<Vec<f32>> {
        let tv = tensors.tensor(key)
            .map_err(|_| anyhow::anyhow!("key '{}' not found in safetensors", key))?;
        let bytes = tv.data();
        anyhow::ensure!(bytes.len() % 4 == 0, "tensor '{}' not f32", key);
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    };

    let mut loaded = 0usize;
    let mut missing = Vec::<String>::new();

    for (key, node_idx, param_idx) in &named {
        // ── Fused Conv2dBnSilu: fold raw BN tensors into precomputed params ──
        if key.ends_with(".bn_scale") {
            let prefix = &key[..key.len() - ".bn_scale".len()];
            match (load_f32(&format!("{prefix}.bn.weight")), load_f32(&format!("{prefix}.bn.running_var"))) {
                (Ok(gamma), Ok(var)) => {
                    let eps = 1e-5f32;
                    let bn_scale: Vec<f32> = gamma.iter().zip(var.iter())
                        .map(|(&g, &v)| g / (v + eps).sqrt()).collect();
                    model.load_param_f32(*node_idx, *param_idx, &bn_scale)
                        .with_context(|| format!("uploading bn_scale for '{key}'"))?;
                    loaded += 1;
                }
                _ => missing.push(key.clone()),
            }
            continue;
        }
        if key.ends_with(".bn_shift") {
            let prefix = &key[..key.len() - ".bn_shift".len()];
            match (
                load_f32(&format!("{prefix}.bn.bias")),
                load_f32(&format!("{prefix}.bn.weight")),
                load_f32(&format!("{prefix}.bn.running_mean")),
                load_f32(&format!("{prefix}.bn.running_var")),
            ) {
                (Ok(beta), Ok(gamma), Ok(mean), Ok(var)) => {
                    let eps = 1e-5f32;
                    let bn_shift: Vec<f32> = beta.iter().zip(gamma.iter()).zip(mean.iter()).zip(var.iter())
                        .map(|(((&b, &g), &m), &v)| b - g / (v + eps).sqrt() * m)
                        .collect();
                    model.load_param_f32(*node_idx, *param_idx, &bn_shift)
                        .with_context(|| format!("uploading bn_shift for '{key}'"))?;
                    loaded += 1;
                }
                _ => missing.push(key.clone()),
            }
            continue;
        }

        // ── Direct lookup with conv.weight fallback for fused nodes ─────────
        let tv_result = match tensors.tensor(key.as_str()) {
            ok @ Ok(_) => ok,
            Err(_) if key.ends_with(".weight") => {
                let prefix = &key[..key.len() - ".weight".len()];
                tensors.tensor(&format!("{prefix}.conv.weight"))
            }
            err => err,
        };

        match tv_result {
            Ok(tv) => {
                let bytes = tv.data();
                anyhow::ensure!(
                    bytes.len() % 4 == 0,
                    "tensor '{key}': byte length not divisible by 4"
                );
                let data: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                model
                    .load_param_f32(*node_idx, *param_idx, &data)
                    .with_context(|| format!("uploading '{key}'"))?;
                loaded += 1;
            }
            Err(_) => missing.push(key.clone()),
        }
    }

    if !missing.is_empty() {
        println!(
            "Warning: {}/{} params not found in safetensors:",
            missing.len(),
            named.len()
        );
        for k in missing.iter().take(10) {
            println!("  missing: {k}");
        }
        if missing.len() > 10 {
            println!("  ... and {} more", missing.len() - 10);
        }
    }
    println!("Loaded {loaded}/{} named parameters.", named.len());
    Ok(())
}

#[cfg(feature = "cuda")]
fn save_checkpoint(
    model: &teeny_cuda::model::LoadedModel,
    param_info: &[(usize, Vec<Vec<usize>>)],
    dir: &Path,
) -> Result<()> {
    use std::io::{BufWriter, Write};
    std::fs::create_dir_all(dir)?;
    let file = std::fs::File::create(dir.join("params.bin"))?;
    let mut w = BufWriter::new(file);
    for (node_idx, shapes) in param_info {
        for (param_idx, _) in shapes.iter().enumerate() {
            let data = model
                .read_param_grad_f32(*node_idx, param_idx)
                .unwrap_or_default();
            let _ = data;
        }
    }
    w.write_all(b"rfdetr-ckpt-v1")?;
    w.flush()?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn restore_checkpoint(
    model: &mut teeny_cuda::model::LoadedModel,
    param_info: &[(usize, Vec<Vec<usize>>)],
    path: &Path,
) -> Result<()> {
    let bytes = std::fs::read(path)?;
    if bytes.starts_with(b"rfdetr-ckpt-v1") {
        println!("Note: checkpoint format does not yet store weights; starting fresh.");
        return Ok(());
    }
    let saved: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let mut cursor = 0usize;
    for (node_idx, shapes) in param_info {
        for (param_idx, shape) in shapes.iter().enumerate() {
            let n: usize = shape.iter().product();
            model.load_param_f32(*node_idx, param_idx, &saved[cursor..cursor + n])?;
            cursor += n;
        }
    }
    Ok(())
}

fn box_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let ax1 = a[0] - a[2] * 0.5;
    let ax2 = a[0] + a[2] * 0.5;
    let ay1 = a[1] - a[3] * 0.5;
    let ay2 = a[1] + a[3] * 0.5;
    let bx1 = b[0] - b[2] * 0.5;
    let bx2 = b[0] + b[2] * 0.5;
    let by1 = b[1] - b[3] * 0.5;
    let by2 = b[1] + b[3] * 0.5;
    let inter = (ax2.min(bx2) - ax1.max(bx1)).max(0.0) * (ay2.min(by2) - ay1.max(by1)).max(0.0);
    let union = a[2] * a[3] + b[2] * b[3] - inter;
    inter / (union + 1e-7)
}

fn compute_ap(preds: &[(f32, bool)], n_gt: usize) -> f32 {
    if n_gt == 0 || preds.is_empty() {
        return 0.0;
    }
    let mut sorted = preds.to_vec();
    sorted.sort_by(|(s1, _), (s2, _)| s2.partial_cmp(s1).unwrap_or(std::cmp::Ordering::Equal));
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut ap = 0.0f32;
    let mut prev_r = 0.0f32;
    for (_, is_tp) in &sorted {
        if *is_tp {
            tp += 1;
        } else {
            fp += 1;
        }
        let r = tp as f32 / n_gt as f32;
        let p = tp as f32 / (tp + fp) as f32;
        if r > prev_r {
            ap += (r - prev_r) * p;
            prev_r = r;
        }
    }
    ap
}
