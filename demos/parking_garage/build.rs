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

// `ui/index.html` is the hand-authored frontend source (Vue/Tailwind loaded from CDN, no
// bundler needed). This just stages it into `ui/dist/index.html`, which
// `src/bin/webapp.rs` pulls in via `include_str!` at compile time. `ui/dist/` is gitignored
// (matched by the repo-root `.gitignore`'s generic `dist/` rule) like any other build output.

use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let src = Path::new(&manifest_dir).join("ui/index.html");
    let dist_dir = Path::new(&manifest_dir).join("ui/dist");
    let dest = dist_dir.join("index.html");

    println!("cargo:rerun-if-changed={}", src.display());

    std::fs::create_dir_all(&dist_dir)
        .unwrap_or_else(|e| panic!("creating {}: {e}", dist_dir.display()));
    std::fs::copy(&src, &dest)
        .unwrap_or_else(|e| panic!("copying {} to {}: {e}", src.display(), dest.display()));
}
