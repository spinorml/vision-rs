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

#![warn(missing_docs)]

//! A high-performance computer vision SDK for Rust, built on the
//! [teenygrad](https://github.com/teenygrad/teenygrad) ML runtime.

/// Object detection interface (detectors, config, results).
pub mod detect;
/// Model architectures and their supporting kernels/loss functions.
pub mod models;
