use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
};
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tower_http::{
    services::{ServeDir, ServeFile},
    trace::TraceLayer,
};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let addr: SocketAddr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:3000".to_owned())
        .parse()
        .context("expected listen address like 127.0.0.1:3000")?;

    let static_files =
        ServeDir::new("ui/dist").not_found_service(ServeFile::new("ui/dist/index.html"));

    let app = Router::new()
        .route("/api/hello", get(hello))
        .route("/api/ws", get(websocket))
        .fallback_service(static_files)
        .layer(TraceLayer::new_for_http());

    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind HTTP listener at {addr}"))?;

    println!("parking garage web client listening on http://{addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("HTTP server failed")?;

    Ok(())
}

#[derive(Serialize)]
struct HelloResponse {
    message: &'static str,
}

async fn hello() -> Json<HelloResponse> {
    Json(HelloResponse {
        message: "hello from parking-garage-client",
    })
}

async fn websocket(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(socket: WebSocket) {
    let (mut sender, mut receiver) = socket.split();

    if sender
        .send(Message::Text("hello from the websocket api".into()))
        .await
        .is_err()
    {
        return;
    }

    while let Some(Ok(message)) = receiver.next().await {
        match message {
            Message::Text(text) => {
                if sender
                    .send(Message::Text(format!("echo: {text}").into()))
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::warn!(%error, "failed to listen for shutdown signal");
    }
}
