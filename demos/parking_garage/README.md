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

The client accepts an optional HTTP listen address. If omitted, it uses `0.0.0.0:3000`.

## Building and Deploying to NVIDIA Jetson (Jetson Orin Nano)

This demo can be cross-compiled and deployed to a Jetson Orin Nano using the same workflow as the main vision-rs crate.

### Prerequisites

- Host system: Ubuntu 24.04+
- Rust toolchain: Installed via [rustup](https://rustup.rs/)
- [cargo-teeny](https://github.com/teenygrad/cargo-teeny): For cross-compiling and packaging
- [cross](https://github.com/cross-rs/cross): For aarch64 cross-builds
- aarch64 CUDA libraries: Downloadable from NVIDIA's CUDA toolkit for Jetson (see README in project root for instructions)
- Jetson device accessible via SSH (for deployment)

### Cross-Compiling for Jetson

On your x86 host machine, ensure `cargo-teeny` and `cross` are installed and the proper target is added:

```bash
cargo install --git https://github.com/teenygrad/cargo-teeny
cargo install cross --git https://github.com/cross-rs/cross
rustup target add aarch64-unknown-linux-gnu
```

#### Package the Demo for Jetson

`cargo teeny` only supports being invoked from the repo root (where `Cross.toml` lives) —
not from within `demos/parking_garage` — so select this crate with `--package`/`-p` instead
of `cd`-ing into its directory:

```bash
# From the root of the repo
source .env
cargo teeny package \
  --target jetson-orin-nano \
  --package parking-garage \
  --bin parking-garage-server \
  --dest ./demos/parking_garage/dist/parking-garage-orin \
  --device cuda \
  --features cuda \
  --options "capability=sm_87,ptx-version=82"
```

This creates a self-contained bundle in `./demos/parking_garage/dist/parking-garage-orin/`:

```text
demos/parking_garage/dist/parking-garage-orin/
  bin/parking-garage-server   # Cross-compiled for Jetson (aarch64)
  cache/                      # Precompiled GPU kernels
  conf/                       # Build provenance info
  data/                       # Empty, will contain models/datasets if used
```

- Adjust `--options` as needed for your Jetson's CUDA version/GPU capability.

#### Deploy to Jetson

Copy the package to your Jetson device using the deploy command (replace user/host as appropriate):

```bash
cargo teeny deploy \
  --package ./demos/parking_garage/dist/parking-garage-orin \
  --host <user>@<orin-host> \
  --dest /home/<user>/parking-garage-demo
```

#### Running on the Jetson

SSH into your Jetson, `cd` into the deployed directory, and run:

```bash
cd /home/<user>/parking-garage-demo
./bin/parking-garage-server
```

If your application uses data/models, place them into the `data/` subfolder as needed.

> For further details on Jetson cross-compilation, packaging, and deployment, see the [vision-rs main README](../../README.md#cross-compilation-for-jetson-orin-nano).
