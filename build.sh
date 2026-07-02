#!/bin/bash
set -e

# Thin wrapper over the CMake build (see CMakeLists.txt / CMakePresets.json).
# Usage:
#   ./build.sh breakout              # Build _C extension with breakout statically linked
#   ./build.sh breakout --float      # float32 precision (required for --slowly)
#   ./build.sh breakout --cpu        # CPU fallback, torch only
#   ./build.sh breakout --debug      # Debug build
#   ./build.sh breakout --local      # Standalone executable (debug, sanitizers)
#   ./build.sh breakout --fast       # Standalone executable (optimized)
#   ./build.sh breakout --web        # Emscripten web build (requires EMSDK)
#   ./build.sh breakout --profile    # Kernel profiling binary
#   ./build.sh all                   # Build all envs with default and --float
#
# On Windows use ./build.ps1 (or the cmake presets directly).

if [ -z "$1" ]; then
    echo "Usage: ./build.sh ENV_NAME [--float] [--debug] [--local|--fast|--web|--profile|--cpu]"
    exit 1
fi
ENV=$1
shift

PRESET=cuda
EXTRA_ARGS=()
for arg in "$@"; do
    case $arg in
        --float)   EXTRA_ARGS+=(-DPRECISION_FLOAT=ON) ;;
        --debug)   DEBUG=1 ;;
        --local)   PRESET=local ;;
        --fast)    PRESET=fast ;;
        --web)     PRESET=web ;;
        --profile) PRESET=profile ;;
        --cpu)     PRESET=cpu ;;
        *) echo "Error: unknown argument '$arg'" && exit 1 ;;
    esac
done

if [ "$PRESET" = "cuda" ] && [ -n "$DEBUG" ]; then
    PRESET=cuda-debug
elif [ -n "$DEBUG" ]; then
    EXTRA_ARGS+=(-DCMAKE_BUILD_TYPE=Debug)
fi

if [ "$ENV" = "all" ]; then
    FAILED=""
    for env_dir in ocean/*/; do
        env=$(basename "$env_dir")
        if bash "$0" "$env" && bash "$0" "$env" --float; then
            echo "OK: $env"
        else
            echo "FAIL: $env"
            FAILED="$FAILED\n  $env"
        fi
    done

    if [ -n "$FAILED" ]; then
        echo -e "\nFailed builds:$FAILED"
    fi
    exit 0
fi

# Legacy env var passthrough: NVCC_ARCH selects CUDA architectures (default native)
if [ -n "$NVCC_ARCH" ]; then
    EXTRA_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$NVCC_ARCH")
fi

cmake --preset "$PRESET" -DENV="$ENV" "${EXTRA_ARGS[@]}"
cmake --build --preset "$PRESET"
