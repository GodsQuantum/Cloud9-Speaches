#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <transcribe.cpp-source-dir> <ubuntu-24.04-build-dir>" >&2
  exit 2
fi

SRC=$(realpath "$1")
BUILD=$(realpath "$2")
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT="$ROOT/cloud9-artifacts/transcribe"

[ -f "$SRC/bindings/python/src/transcribe_cpp/__init__.py" ] || { echo "Invalid transcribe.cpp source: $SRC" >&2; exit 1; }
[ -f "$BUILD/src/libtranscribe.so.0.2.3" ] || { echo "Invalid Noble shared build: $BUILD" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT/lib" "$OUT/python"
cp -a "$BUILD"/src/libtranscribe.so* "$OUT/lib/"
cp -a "$BUILD"/ggml/src/libggml.so* "$OUT/lib/"
cp -a "$BUILD"/ggml/src/libggml-base.so* "$OUT/lib/"
cp -a "$BUILD"/ggml/src/libggml-cpu.so* "$OUT/lib/"
cp -a "$BUILD"/ggml/src/ggml-vulkan/libggml-vulkan.so* "$OUT/lib/"
cp -a "$SRC/bindings/python/src/transcribe_cpp" "$OUT/python/"
find "$OUT/python" -type d -name __pycache__ -prune -exec rm -rf {} +

echo "Staged transcribe.cpp runtime artifacts in $OUT"
