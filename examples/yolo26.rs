/*
 * SpinorML Ltd 🚀 AGPL-3.0 License - https://spinorml.com/license
 */

//! Download and view a vision-rs dataset from a model config TOML.
//!
//! Usage:
//!   cargo run --example yolo26 -- download --model assets/models/coco128.toml
//!   cargo run --example yolo26 -- view     --model assets/models/coco128.toml

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dotenv::dotenv;
use eframe::egui;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Download and view a vision-rs model dataset")]
struct Args {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download and cache a dataset from a model config TOML
    Download {
        #[arg(short, long)]
        model: PathBuf,
    },
    /// View training images with bounding boxes
    View {
        #[arg(short, long)]
        model: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Model config (model TOML)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ModelConfig {
    dataset: DatasetMeta,
}

#[derive(Deserialize)]
struct DatasetMeta {
    name: String,
    url:  String,
}

// ---------------------------------------------------------------------------
// Labels file (train/labels.toml)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct LabelsFile {
    classes: ClassesMeta,
    #[serde(default)]
    images: Vec<ImageEntry>,
}

#[derive(Deserialize)]
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
    bbox: [f32; 4], // [cx, cy, w, h] normalised to [0, 1]
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    dotenv().ok();
    let args = Args::parse();

    match args.command {
        Cmd::Download { model } => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(run_download(model)),
        Cmd::View { model } => run_view(model),
    }
}

// ---------------------------------------------------------------------------
// Download command
// ---------------------------------------------------------------------------

async fn run_download(model: PathBuf) -> Result<()> {
    let config_text = std::fs::read_to_string(&model)
        .with_context(|| format!("reading model config {:?}", model))?;
    let config: ModelConfig = toml::from_str(&config_text)
        .context("parsing model config TOML")?;

    let cache_dir: PathBuf = std::env::var("DATASETS_CACHE_DIR")
        .context("DATASETS_CACHE_DIR not set — add it to .env")?
        .into();

    let dest = cache_dir.join(&config.dataset.name);
    if dest.exists() {
        println!("already cached at {}", dest.display());
        return Ok(());
    }

    tokio::fs::create_dir_all(&cache_dir).await
        .with_context(|| format!("creating cache dir {}", cache_dir.display()))?;

    let zip_path = download(&config.dataset.name, &config.dataset.url, &cache_dir).await?;
    extract(&config.dataset.name, zip_path, cache_dir).await?;

    println!("\n{} ready at {}", config.dataset.name, dest.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// View command
// ---------------------------------------------------------------------------

fn run_view(model: PathBuf) -> Result<()> {
    let config_text = std::fs::read_to_string(&model)
        .with_context(|| format!("reading model config {:?}", model))?;
    let config: ModelConfig = toml::from_str(&config_text)
        .context("parsing model config TOML")?;

    let cache_dir: PathBuf = std::env::var("DATASETS_CACHE_DIR")
        .context("DATASETS_CACHE_DIR not set — add it to .env")?
        .into();

    let dataset_dir  = cache_dir.join(&config.dataset.name);
    let labels_path  = dataset_dir.join("train").join("labels.toml");
    let images_dir   = dataset_dir.join("train").join("images");

    let labels_text = std::fs::read_to_string(&labels_path)
        .with_context(|| format!("reading {:?}", labels_path))?;
    let labels: LabelsFile = toml::from_str(&labels_text)
        .context("parsing labels.toml")?;

    if labels.images.is_empty() {
        anyhow::bail!("no images in {:?}", labels_path);
    }

    let title = format!(
        "{} — train ({} images)",
        config.dataset.name,
        labels.images.len()
    );

    let app = ViewApp::new(labels.images, labels.classes.names, images_dir);

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
    entries:    Vec<ImageEntry>,
    classes:    Vec<String>,
    idx:        usize,
    jump_buf:   String,
    texture:    Option<egui::TextureHandle>,
    loaded_idx: Option<usize>,
}

impl ViewApp {
    fn new(entries: Vec<ImageEntry>, classes: Vec<String>, images_dir: PathBuf) -> Self {
        Self {
            images_dir,
            entries,
            classes,
            idx: 0,
            jump_buf: String::new(),
            texture: None,
            loaded_idx: None,
        }
    }

    fn prev(&mut self) { if self.idx > 0 { self.idx -= 1; } }

    fn next(&mut self) { if self.idx + 1 < self.entries.len() { self.idx += 1; } }

    fn jump(&mut self, one_based: usize) {
        self.idx = one_based.saturating_sub(1).min(self.entries.len().saturating_sub(1));
    }

    fn load_texture(&mut self, ctx: &egui::Context) {
        let path = self.images_dir.join(&self.entries[self.idx].file);
        self.loaded_idx = Some(self.idx);

        match image::open(&path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &rgba);
                self.texture = Some(ctx.load_texture(
                    "dataset-image",
                    color_image,
                    egui::TextureOptions::LINEAR,
                ));
            }
            Err(e) => {
                eprintln!("failed to load {:?}: {e}", path);
                self.texture = None;
            }
        }
    }
}

impl eframe::App for ViewApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Keyboard navigation
        ctx.input(|i| {
            if i.key_pressed(egui::Key::ArrowLeft)  { self.prev(); }
            if i.key_pressed(egui::Key::ArrowRight) { self.next(); }
        });

        if self.loaded_idx != Some(self.idx) {
            self.load_texture(ctx);
        }

        egui::TopBottomPanel::bottom("nav").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("⬅ Prev").clicked() { self.prev(); }
                ui.label(format!("{} / {}", self.idx + 1, self.entries.len()));
                if ui.button("Next ➡").clicked() { self.next(); }

                ui.separator();
                ui.label("Go to:");
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut self.jump_buf).desired_width(56.0),
                );
                if resp.lost_focus() && ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if let Ok(n) = self.jump_buf.trim().parse::<usize>() {
                        self.jump(n);
                    }
                }

                ui.separator();
                let entry = &self.entries[self.idx];
                ui.label(format!(
                    "{}  ({} boxes)",
                    entry.file,
                    entry.annotations.len()
                ));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(texture) = &self.texture {
                let tex_size  = texture.size_vec2();
                let available = ui.available_size();
                let scale     = (available.x / tex_size.x).min(available.y / tex_size.y);
                let display   = tex_size * scale;

                let (rect, _) = ui.allocate_exact_size(display, egui::Sense::hover());
                let painter   = ui.painter();

                painter.image(
                    texture.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );

                let annotations = self.entries[self.idx].annotations.clone();
                for ann in &annotations {
                    let [cx, cy, bw, bh] = ann.bbox;
                    let x1 = rect.left() + (cx - bw * 0.5) * rect.width();
                    let y1 = rect.top()  + (cy - bh * 0.5) * rect.height();
                    let x2 = rect.left() + (cx + bw * 0.5) * rect.width();
                    let y2 = rect.top()  + (cy + bh * 0.5) * rect.height();
                    let box_rect = egui::Rect::from_min_max(
                        egui::pos2(x1, y1),
                        egui::pos2(x2, y2),
                    );
                    let color = class_color(ann.class_id);

                    painter.rect_stroke(box_rect, 0.0, egui::Stroke::new(2.0, color));

                    let label = self
                        .classes
                        .get(ann.class_id)
                        .map(|s| s.as_str())
                        .unwrap_or("?");
                    let galley = painter.layout_no_wrap(
                        label.to_string(),
                        egui::FontId::proportional(12.0),
                        egui::Color32::WHITE,
                    );
                    let label_size = galley.size() + egui::vec2(4.0, 2.0);
                    let label_origin = egui::pos2(x1, (y1 - label_size.y).max(rect.top()));
                    let bg = egui::Rect::from_min_size(label_origin, label_size);
                    painter.rect_filled(bg, 2.0, color);
                    painter.galley(label_origin + egui::vec2(2.0, 1.0), galley, egui::Color32::WHITE);
                }
            } else {
                ui.centered_and_justified(|ui| { ui.label("Loading…"); });
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Color palette — golden-angle hue step, full saturation / value
// ---------------------------------------------------------------------------

fn class_color(class_id: usize) -> egui::Color32 {
    let hue = (class_id as f32 * 137.508) % 360.0;
    let h   = hue / 60.0;
    let x   = 1.0 - (h % 2.0 - 1.0).abs();
    let (r, g, b) = match h as u32 {
        0 => (1.0_f32, x,   0.0),
        1 => (x,       1.0, 0.0),
        2 => (0.0,     1.0, x  ),
        3 => (0.0,     x,   1.0),
        4 => (x,       0.0, 1.0),
        _ => (1.0,     0.0, x  ),
    };
    egui::Color32::from_rgb(
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
    )
}

// ---------------------------------------------------------------------------
// Download helpers
// ---------------------------------------------------------------------------

async fn download(name: &str, url: &str, cache_dir: &Path) -> Result<PathBuf> {
    let resp = Client::builder()
        .user_agent("vision-rs")
        .build()?
        .get(url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {url}"))?;

    let total    = resp.content_length().unwrap_or(0);
    let zip_path = cache_dir.join(format!("{name}.zip"));

    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg}\n  [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta})")?
            .progress_chars("█▉▊▋▌▍▎▏  "),
    );
    pb.set_message(format!("Downloading {name}"));

    let mut file   = tokio::fs::File::create(&zip_path).await?;
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading response chunk")?;
        pb.inc(chunk.len() as u64);
        file.write_all(&chunk).await.context("writing to disk")?;
    }
    pb.finish_with_message(format!("Downloaded  {name}"));

    Ok(zip_path)
}

async fn extract(name: &str, zip_path: PathBuf, dest_dir: PathBuf) -> Result<()> {
    let name = name.to_owned();
    tokio::task::spawn_blocking(move || -> Result<()> {
        let file    = std::fs::File::open(&zip_path)?;
        let mut arc = zip::ZipArchive::new(file)?;

        let pb = ProgressBar::new(arc.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg}\n  [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} files ({per_sec}, eta {eta})")?
                .progress_chars("█▉▊▋▌▍▎▏  "),
        );
        pb.set_message(format!("Extracting  {name}"));

        for i in 0..arc.len() {
            let mut entry = arc.by_index(i)?;
            let out = match entry.enclosed_name() {
                Some(p) => dest_dir.join(p),
                None    => continue,
            };
            if entry.is_dir() {
                std::fs::create_dir_all(&out)?;
            } else {
                if let Some(p) = out.parent() { std::fs::create_dir_all(p)?; }
                std::io::copy(&mut entry, &mut std::fs::File::create(&out)?)?;
            }
            pb.inc(1);
        }

        pb.finish_with_message(format!("Extracted   {name}"));
        std::fs::remove_file(&zip_path)?;
        Ok(())
    })
    .await
    .context("extraction task panicked")??;

    Ok(())
}
