#!/usr/bin/env bash
# Build and run gemm (im2col+cuBLAS) vs cuDNN conv benchmark.
# Default: fp32. Use --bf16 / --half for bf16 (matches native backend without --float).
#
# im2col vs im2col_kernel_fast (correctness + timing), same layer sizes as conv bench:
#   ./tests/bench_conv_gemm_vs_cudnn.sh --layer 1 --im2col-bench-only
#   ./tests/bench_conv_gemm_vs_cudnn.sh --bf16 --layer 2 --im2col-bench
# gemm slow vs fast forward (relu 0/1), NMMO3 layer sizes:
#   ./tests/bench_conv_gemm_vs_cudnn.sh --layer 1 --gemm-fast-bench-only
# ∂W-only vs full backward (gemm vs cudnn):
#   ./tests/bench_conv_gemm_vs_cudnn.sh --layer 1 --bwd-dinput-bench-only
# gemm vs gemm_fast vs cudnn full backward:
#   ./tests/bench_conv_gemm_vs_cudnn.sh --layer 1 --gemm-bwd-fast-bench-only
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-$(dirname "$(dirname "$(command -v nvcc)")")}}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
ARCH="${NVCC_ARCH:-native}"

# Default fp32; --bf16 / --half drop -DPRECISION_FLOAT
PRECISION_FLAG="-DPRECISION_FLOAT"
USER_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --bf16|--half) PRECISION_FLAG="" ;;
    --float|--fp32) PRECISION_FLAG="-DPRECISION_FLOAT" ;;
    *) USER_ARGS+=("$arg") ;;
  esac
done

CUDNN_IFLAG=""
CUDNN_LFLAG=""
for dir in "$CUDA_HOME/include" /usr/local/cuda/include /usr/include; do
  if [[ -f "$dir/cudnn.h" ]]; then
    CUDNN_IFLAG="-I$dir"
    break
  fi
done
for dir in "$CUDA_HOME/lib64" "$CUDA_HOME/lib" /usr/lib/x86_64-linux-gnu; do
  if [[ -f "$dir/libcudnn.so" ]] || [[ -f "$dir/libcudnn.dylib" ]]; then
    CUDNN_LFLAG="-L$dir"
    break
  fi
done
if [[ -z "$CUDNN_IFLAG" ]]; then
  CUDNN_IFLAG=$(python3 -c "import nvidia.cudnn, os; print('-I' + os.path.join(nvidia.cudnn.__path__[0], 'include'))" 2>/dev/null || true)
fi
if [[ -z "$CUDNN_LFLAG" ]]; then
  CUDNN_LFLAG=$(python3 -c "import nvidia.cudnn, os; print('-L' + os.path.join(nvidia.cudnn.__path__[0], 'lib'))" 2>/dev/null || true)
fi

OUT="${ROOT}/tests/bench_conv_gemm_vs_cudnn"
if [[ -n "$PRECISION_FLAG" ]]; then
  echo "nvcc $ARCH $OUT  (fp32)"
else
  echo "nvcc $ARCH $OUT  (bf16)"
fi
"$NVCC" -O2 -std=c++17 "-arch=$ARCH" \
  "-I${ROOT}/src" \
  "-I${CUDA_HOME}/include" \
  $CUDNN_IFLAG \
  $PRECISION_FLAG \
  "${ROOT}/tests/bench_conv_gemm_vs_cudnn.cu" \
  -o "$OUT" \
  $CUDNN_LFLAG \
  -L"${CUDA_HOME}/lib64" -L"${CUDA_HOME}/lib" \
  -lcublas -lcudnn -lcurand

echo "Running: $OUT ${USER_ARGS[*]}"
exec "$OUT" "${USER_ARGS[@]}"
