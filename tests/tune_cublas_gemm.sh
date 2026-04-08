#!/usr/bin/env bash
# Build and run cublasGemmEx tuner (same layout as puf_mm_tn in kernels.cu).
# Default: fp32. Use --bf16 / --half for bf16.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-$(dirname "$(dirname "$(command -v nvcc)")")}}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
ARCH="${NVCC_ARCH:-native}"

PRECISION_FLAG="-DPRECISION_FLOAT"
USER_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --bf16|--half) PRECISION_FLAG="" ;;
    --float|--fp32) PRECISION_FLAG="-DPRECISION_FLOAT" ;;
    *) USER_ARGS+=("$arg") ;;
  esac
done

OUT="${ROOT}/tests/tune_cublas_gemm"
if [[ -n "$PRECISION_FLAG" ]]; then
  echo "nvcc $ARCH $OUT  (fp32)"
else
  echo "nvcc $ARCH $OUT  (bf16)"
fi
"$NVCC" -O2 -std=c++17 "-arch=$ARCH" \
  $PRECISION_FLAG \
  "${ROOT}/tests/tune_cublas_gemm.cu" \
  -o "$OUT" \
  -L"${CUDA_HOME}/lib64" -L"${CUDA_HOME}/lib" \
  -lcublas

echo "Running: $OUT ${USER_ARGS[*]}"
exec "$OUT" "${USER_ARGS[@]}"
