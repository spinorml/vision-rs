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

/*
 * Static file server for the parking garage frontend.
 *
 * Ports:
 *   webapp (this binary) — default 3000  — serves ui/dist/
 *   server               — default 3001  — WebSocket API (ws://localhost:3001/api/ws)
 */

use anyhow::{Context, Result};
use axum::Router;
use dotenv::dotenv;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tower_http::{
    services::{ServeDir, ServeFile},
    trace::TraceLayer,
};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    // `cargo teeny package --bin parking-garage-webapp ...` drives every packaged binary
    // through the same host-side AOT step (`cargo run --bin <name> -- --device ... --options
    // ...`, see `parking-garage-server`'s `is_aot_invocation`/`run_aot`). This binary has no
    // GPU kernels to compile, so just no-op instead of misparsing `--device`'s value as the
    // listen address below.
    if std::env::args().any(|a| a == "--device") {
        println!("parking-garage-webapp: nothing to AOT-compile (no GPU kernels)");
        return Ok(());
    }

    tracing_subscriber::fmt::init();

    let addr: SocketAddr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "0.0.0.0:3000".to_owned())
        .parse()
        .context("expected listen address like 127.0.0.1:3000")?;

    let static_files =
        ServeDir::new("ui/dist").not_found_service(ServeFile::new("ui/dist/index.html"));

    let app = Router::new()
        .fallback_service(static_files)
        .layer(TraceLayer::new_for_http());

    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind HTTP listener at {addr}"))?;

    println!("parking garage webapp on http://{addr}");
    println!("  → frontend connects to ws://localhost:3001/api/ws by default");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("HTTP server failed")?;

    Ok(())
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::warn!(%error, "failed to listen for shutdown signal");
    }
}
