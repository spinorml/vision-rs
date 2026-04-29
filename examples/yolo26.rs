/*
 * SpinorML Ltd 🚀 AGPL-3.0 License - https://spinorml.com/license
 */

//! Download and cache a vision-rs dataset from a model config TOML.
//!
//! Usage:
//!   cargo run --example yolo26 -- --config assets/models/coco128.toml

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use dotenv::dotenv;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Download and cache a vision-rs model dataset")]
struct Args {
    /// Path to the model config TOML (e.g. assets/models/coco128.toml)
    #[arg(short, long)]
    config: PathBuf,
}

// ---------------------------------------------------------------------------
// Config shape (only the fields we need from the TOML)
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
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    let args = Args::parse();

    let config_text = std::fs::read_to_string(&args.config)
        .with_context(|| format!("reading config {:?}", args.config))?;
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
// Download
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

    let total = resp.content_length().unwrap_or(0);
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

// ---------------------------------------------------------------------------
// Extract
// ---------------------------------------------------------------------------

async fn extract(name: &str, zip_path: PathBuf, dest_dir: PathBuf) -> Result<()> {
    // Zip extraction is synchronous and CPU-bound; run it off the async executor.
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
