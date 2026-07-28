# Introduction

**vision-rs** is a high-performance computer vision SDK for Rust, built on
[teenygrad](https://github.com/teenygrad/teenygrad), a memory-safe Rust ML
training and inference runtime. It currently ships **YOLO26**, a real-time
object detection model family, with custom GPU kernels (attention, detection
decode, loss functions) written directly against teenygrad's Triton-style
kernel DSL.

## What this book covers

- Getting a working detector running against a JPEG/PNG image.
- The `ObjectDetector`/`DetectorConfig` API surface.
- How the YOLO26 model is assembled from backbone/neck/head blocks, and how
  the N/S/M/L/XL variants differ.
- Training: the TaskAlignedAssigner, CIoU/classification losses, and the
  dual-head (one2many/one2one) assignment scheme.
- The custom kernels vision-rs ships (Flash Attention 2, position-sensitive
  attention, detect-decode) and the `teenyc` toolchain that compiles them.
- Cross-compiling and packaging a self-contained deployable bundle for a
  Jetson Orin Nano.

## What this book does not cover

This book is a guide to *using* vision-rs. For the full API reference
(every public type, field, and function), see the
[crate documentation](../api/vision_rs/index.html). For the ML runtime
vision-rs is built on — tensors, the computational graph, the kernel
compiler — see [The Teenygrad Book](https://docs.teenygrad.org/book/introduction.html).

## Status

vision-rs is under active development. The public API (`vision_rs::detect`)
is the intended stable surface; everything under `vision_rs::models` is
expected to grow additional model families over time. See the
[FAQ & Roadmap](./appendix/faq-and-roadmap.md) for what's implemented today.
