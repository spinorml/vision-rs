# Parking Garage Demo

This crate demonstrates the two-process parking garage shape:

- `parking-garage-server` publishes sample backend events over a Unix domain socket.
- `parking-garage-webapp` serves a small Vue page and exposes an HTTP/WebSocket API for the browser.

Run the server:

```bash
cargo run --bin parking-garage-server
```

Run the browser-facing webapp in another terminal:

```bash
cargo run --bin parking-garage-webapp
```

Then open <http://127.0.0.1:3000>.

The server accepts an optional Unix socket path. If omitted, it uses `/tmp/vision-rs-parking-garage.sock`.

The client accepts an optional HTTP listen address. If omitted, it uses `127.0.0.1:3000`.
