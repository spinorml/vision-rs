use anyhow::{Context, Result};
use futures_util::SinkExt;
use parking_garage::{DEFAULT_SOCKET_PATH, ParkingEvent, encode_event};
use tokio::net::UnixListener;
use tokio::time::{Duration, interval};
use tokio_util::codec::{Framed, LengthDelimitedCodec};

#[tokio::main]
async fn main() -> Result<()> {
    let socket_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_SOCKET_PATH.to_owned());

    remove_stale_socket(&socket_path).await?;

    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("failed to bind Unix socket at {socket_path}"))?;

    println!("parking garage server listening on {socket_path}");

    loop {
        let (stream, _addr) = listener.accept().await.context("accept failed")?;
        println!("client connected");

        if let Err(error) = publish_events(stream).await {
            eprintln!("client disconnected: {error:#}");
        }
    }
}

async fn remove_stale_socket(socket_path: &str) -> Result<()> {
    match tokio::fs::remove_file(socket_path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("failed to remove stale socket {socket_path}")),
    }
}

async fn publish_events(stream: tokio::net::UnixStream) -> Result<()> {
    let mut framed = Framed::new(stream, LengthDelimitedCodec::new());
    let mut ticks = interval(Duration::from_secs(1));
    let mut sequence = 0_u64;

    loop {
        ticks.tick().await;
        sequence += 1;

        let event = match sequence % 3 {
            0 => ParkingEvent::VehicleSeen {
                id: sequence,
                bay: format!("A-{}", sequence % 12),
            },
            1 => ParkingEvent::Occupancy {
                occupied: (sequence % 12) as u32,
                capacity: 12,
            },
            _ => ParkingEvent::Heartbeat { sequence },
        };

        framed
            .send(encode_event(&event).context("failed to encode event")?)
            .await
            .context("failed to send event")?;

        println!("published {event:?}");
    }
}
