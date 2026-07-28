#!/bin/sh
# Builds the self-hosted docs site (API reference + mdBook) into
# docs-site/dist/, ready to be served as static files (see Dockerfile).
#
# Requires `mdbook`/`mdbook-mermaid` (`cargo install mdbook mdbook-mermaid`,
# then `mdbook-mermaid install book` once, already done in this repo). The
# CUDA toolkit is optional here (unlike teenygrad's teeny-cuda, vision-rs's
# cuda/training features are just Cargo features, not a hard build.rs
# requirement) -- see the doc_features fallback below.
set -e

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$repo_root"

dist="docs-site/dist"
rm -rf "$dist"
mkdir -p "$dist/api"

# target/doc accumulates output from whatever was last built there --
# clean it first so we only publish exactly what this run produces.
rm -rf target/doc

# Full docs (cuda + training feature-gated items included) need the CUDA
# toolkit on the host, since the cuda feature pulls in teeny-cuda's build.rs
# (hard bindgen dependency on real CUDA headers). Fall back to
# --no-default-features -- same as this crate's docs.rs config -- if it's
# not available, rather than failing the whole build.
if command -v nvcc >/dev/null 2>&1; then
  doc_features="--all-features"
else
  echo "docs-site/build.sh: nvcc not found -- building docs with --no-default-features" >&2
  doc_features="--no-default-features"
fi

# shellcheck disable=SC2086
cargo doc --no-deps $doc_features

cp -r target/doc/. "$dist/api/"

# cargo doc doesn't generate a crate index at the api/ root -- redirect
# straight to the crate's own index instead of building a synthetic list
# (there's only one crate here, unlike teenygrad's workspace).
echo '<!doctype html><meta charset=utf-8><meta http-equiv=refresh content="0; url=vision_rs/index.html">' \
  > "$dist/api/index.html"

# mdBook SDK book.
( cd book && mdbook build )
cp -r book/book "$dist/book"

cp docs-site/index.html "$dist/index.html"

echo "Docs site built at $dist"
