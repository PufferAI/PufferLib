#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

OUT="${TMPDIR:-/tmp}/affine_lock_cuda_tests"
CUDA_ROOT="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
NVCC_BIN="${NVCC:-$CUDA_ROOT/bin/nvcc}"
CUDA_ARCH="${NVCC_ARCH:-native}"

RAYLIB_ROOT="$ROOT/raylib-5.5_linux_amd64"
if [ ! -d "$RAYLIB_ROOT/include" ]; then
    echo "raylib-5.5_linux_amd64 not found" >&2
    exit 1
fi

"$NVCC_BIN" \
    -std=c++17 -O3 -lineinfo -arch="$CUDA_ARCH" \
    -Xcompiler=-Wall,-Wextra,-Werror,-Wno-unused-function,-Wno-unused-parameter,-Wno-missing-field-initializers \
    -Xcompiler=-ffunction-sections,-fdata-sections \
    -I"$ROOT" -I"$ROOT/src" -I"$ROOT/ocean/affine_lock" \
    -I"$ROOT/vendor" -I"$RAYLIB_ROOT/include" \
    "$ROOT/ocean/affine_lock/tests/test_affine_lock_cuda.cu" \
    "$RAYLIB_ROOT/lib/libraylib.a" \
    -Xlinker=--gc-sections \
    -lGL -lpthread -ldl -lrt -lm \
    -o "$OUT"

"$OUT"
