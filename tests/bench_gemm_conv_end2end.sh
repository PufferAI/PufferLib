#!/usr/bin/env bash
# Build and run end-to-end conv benchmark: gemm vs gemm_fast vs cudnn (fwd+bwd), layers 1 & 2,
# plus NMMO3 microbenches (each optional via flags; default runs conv + multihot + embedding).
# Args:
#   --float | --fp32 (default) or --bf16 | --half
#   --multihot-only   n3_multihot ref vs fast only (B=1024..32768)
#   --embedding-only  n3_embedding ref vs fast only (same B grid)
#
#   ./tests/bench_gemm_conv_end2end.sh
#   ./tests/bench_gemm_conv_end2end.sh --bf16
#   ./tests/bench_gemm_conv_end2end.sh --multihot-only
#   ./tests/bench_gemm_conv_end2end.sh --embedding-only
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-$(dirname "$(dirname "$(command -v nvcc)")")}}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
ARCH="${NVCC_ARCH:-native}"

PRECISION_FLAG="-DPRECISION_FLOAT"
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --bf16|--half) PRECISION_FLAG="" ;;
    --float|--fp32) PRECISION_FLAG="-DPRECISION_FLOAT" ;;
    --multihot-only|--embedding-only) EXTRA_ARGS+=("$arg") ;;
    *)
      echo "Unknown argument: $arg (use --float, --bf16, --multihot-only, or --embedding-only)" >&2
      exit 1
      ;;
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

OUT="${ROOT}/tests/bench_gemm_conv_end2end"
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
  "${ROOT}/tests/bench_gemm_conv_end2end.cu" \
  -o "$OUT" \
  $CUDNN_LFLAG \
  -L"${CUDA_HOME}/lib64" -L"${CUDA_HOME}/lib" \
  -lcublas -lcudnn -lcurand

exec "$OUT" "${EXTRA_ARGS[@]}"
