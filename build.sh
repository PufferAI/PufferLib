#!/bin/bash
set -e

# Usage:
#   ./build.sh breakout              # Full native train/eval binary
#   ./build.sh breakout --float      # float32 precision (required for --slowly)
#   ./build.sh breakout --cpu        # Tiny standalone CPU eval executable
#   ./build.sh breakout --debug      # Debug build
#   ./build.sh breakout --local      # Standalone executable (debug, sanitizers)
#   ./build.sh breakout --fast       # Standalone executable (optimized)
#   ./build.sh breakout --web        # Emscripten web build
#   ./build.sh breakout --profile    # Kernel profiling binary
#   ./build.sh all                   # Build all envs native and native float32

if [ -z "$1" ]; then
    echo "Usage: ./build.sh ENV_NAME [--float] [--debug] [--local|--fast|--web|--profile|--cpu]"
    exit 1
fi
ENV=$1
shift

for arg in "$@"; do
    case $arg in
        --float) PRECISION="-DPRECISION_FLOAT" ;;
        --debug) DEBUG=1 ;;
        --local) MODE=local ;;
        --fast)  MODE=fast ;;
        --web)   MODE=web ;;
        --profile) MODE=profile ;;
        --cpu)   MODE=cpu ;;
        *) echo "Error: unknown argument '$arg'" && exit 1 ;;
    esac
done

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

# Linux/mac
PLATFORM="$(uname -s)"
if [ "$PLATFORM" = "Linux" ]; then
    RAYLIB_NAME='raylib-5.5_linux_amd64'
    OMP_LIB=-lomp5
    SANITIZE_FLAGS=(-fsanitize=address,undefined,bounds,pointer-overflow,leak -fno-omit-frame-pointer)
    STANDALONE_LDFLAGS=(-lGL)
else
    RAYLIB_NAME='raylib-5.5_macos'
    OMP_LIB=-lomp
    SANITIZE_FLAGS=()
    STANDALONE_LDFLAGS=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
fi

CLANG_WARN=(
    -Wall
    -ferror-limit=3
    -Werror=incompatible-pointer-types
    -Werror=return-type
    -Wno-error=incompatible-pointer-types-discards-qualifiers
    -Wno-incompatible-pointer-types-discards-qualifiers
    -Wno-error=array-parameter
)

download() {
    local name=$1 url=$2
    [ -d "$name" ] && return
    echo "Downloading $name..."
    case "$url" in
        *.zip) curl -sL "$url" -o "$name.zip" && unzip -q "$name.zip" && rm "$name.zip" ;;
        *)     curl -sL "$url" -o "$name.tar.gz" && tar xf "$name.tar.gz" && rm "$name.tar.gz" ;;
    esac
}

RAYLIB_URL="https://github.com/raysan5/raylib/releases/download/5.5"
if [ "$MODE" = "web" ]; then
    RAYLIB_NAME='raylib-5.5_webassembly'
    download "$RAYLIB_NAME" "$RAYLIB_URL/$RAYLIB_NAME.zip"
else
    download "$RAYLIB_NAME" "$RAYLIB_URL/$RAYLIB_NAME.tar.gz"
fi

RAYLIB_A="$RAYLIB_NAME/lib/libraylib.a"
INCLUDES=(-I./$RAYLIB_NAME/include -I./src -I./vendor)
LINK_ARCHIVES=("$RAYLIB_A")
EXTRA_SRC=""
EXTRA_LDFLAGS=()
EXTRA_CFLAGS=()
SRC_FILE=""

if [ "$ENV" = "constellation" ]; then
    SRC_DIR="src"
    OUTPUT_NAME="seethestars"
    MODE=${MODE:-fast}
    CLANG_WARN+=(-Wno-unused-function)
elif [ "$ENV" = "cache_data" ]; then
    SRC_DIR="src"
    OUTPUT_NAME="cache_data"
    SRC_FILE="src/constellation.c"
    EXTRA_CFLAGS+=(-DPUFFER_CACHE_DATA)
    MODE=${MODE:-fast}
    CLANG_WARN+=(-Wno-unused-function)
elif [ "$ENV" = "trailer" ]; then
    SRC_DIR="trailer"
    OUTPUT_NAME="trailer/trailer"
elif [ "$ENV" = "impulse_wars" ]; then
    SRC_DIR="ocean/$ENV"
    if [ "$MODE" = "web" ]; then BOX2D_NAME='box2d-web'
    elif [ "$PLATFORM" = "Linux" ]; then BOX2D_NAME='box2d-linux-amd64'
    else BOX2D_NAME='box2d-macos-arm64'
    fi
    BOX2D_URL="https://github.com/capnspacehook/box2d/releases/latest/download"
    download "$BOX2D_NAME" "$BOX2D_URL/$BOX2D_NAME.tar.gz"
    INCLUDES+=(-I./$BOX2D_NAME/include -I./$BOX2D_NAME/src)
    LINK_ARCHIVES+=("./$BOX2D_NAME/libbox2d.a")
elif [ "$ENV" = "nethack" ]; then
    SRC_DIR="ocean/$ENV"
    NLE_DIR="vendor/nle"
    NLE_REPO="https://github.com/liujonathan24/NetHack.git"
    if [ ! -d "$NLE_DIR/src" ]; then
        echo "Cloning modified NLE from $NLE_REPO ..."
        git clone --depth 1 "$NLE_REPO" "$NLE_DIR"
    fi
    NETHACK_LIB_DIR="$(pwd)/$NLE_DIR/src/build"
    if [ ! -f "$NETHACK_LIB_DIR/libnethack.so" ]; then
        echo "Building libnethack.so ..."
        make -C "$NETHACK_LIB_DIR" nethack -j$(nproc)
    fi
    INCLUDES+=(-I./$NLE_DIR/include)
    EXTRA_LDFLAGS+=(-L"$NETHACK_LIB_DIR" -lnethack -Wl,-rpath,"$NETHACK_LIB_DIR" -ldl)
elif [ -d "ocean/$ENV" ]; then
    SRC_DIR="ocean/$ENV"
else
    echo "Error: environment '$ENV' not found" && exit 1
fi

OUTPUT_NAME=${OUTPUT_NAME:-$ENV}
SRC_FILE=${SRC_FILE:-$SRC_DIR/$ENV.c}

# Standalone environment build
# -mavx2 enables AVX2 intrinsics (__m256, _mm256_*) which drive.h and
# src/pufferenv.h use directly. x86_64 only — strip if porting to ARM/Apple Silicon.
SIMD_FLAGS=(-mavx2 -mfma)
if [ -n "$DEBUG" ] || [ "$MODE" = "local" ]; then
    CLANG_OPT=(-g -O0 "${CLANG_WARN[@]}" "${SANITIZE_FLAGS[@]}" "${SIMD_FLAGS[@]}")
    NVCC_OPT="-O0 -g"
    LINK_OPT="-g"
else
    CLANG_OPT=(-O2 -DNDEBUG "${CLANG_WARN[@]}" "${SIMD_FLAGS[@]}")
    NVCC_OPT="-O2 --threads 0"
    LINK_OPT="-O2"
fi
if [ "$MODE" = "local" ] || [ "$MODE" = "fast" ]; then
    FLAGS=(
        "${INCLUDES[@]}"
        "$SRC_FILE" $EXTRA_SRC -o "$OUTPUT_NAME"
        "${LINK_ARCHIVES[@]}"
        "${EXTRA_LDFLAGS[@]}"
        "${STANDALONE_LDFLAGS[@]}"
        -lm -lpthread -fopenmp
        -DPLATFORM_DESKTOP
        "${EXTRA_CFLAGS[@]}"
    )
    echo "Compiling $ENV..."
    ${CC:-clang} "${CLANG_OPT[@]}" "${FLAGS[@]}"
    echo "Built: ./$OUTPUT_NAME"
    exit 0
elif [ "$MODE" = "web" ]; then
    mkdir -p "build/web/$ENV"
    echo "Compiling $ENV for web..."
    emcc \
        -o "build/web/$ENV/game.html" \
        "$SRC_FILE" $EXTRA_SRC \
        -O3 -Wall \
        "${LINK_ARCHIVES[@]}" \
        "${INCLUDES[@]}" \
        -L. -L./$RAYLIB_NAME/lib \
        -sASSERTIONS=2 -gsource-map \
        -sUSE_GLFW=3 -sUSE_WEBGL2=1 -sASYNCIFY -sFILESYSTEM -sFORCE_FILESYSTEM=1 \
        --shell-file vendor/minshell.html \
        -sINITIAL_MEMORY=512MB -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=512KB \
        -DNDEBUG -DPLATFORM_WEB -DGRAPHICS_API_OPENGL_ES3 \
        --preload-file resources/$ENV@resources/$ENV \
        --preload-file resources/shared@resources/shared \
        "${EXTRA_CFLAGS[@]}"
    echo "Built: build/web/$ENV/game.html"
    exit 0
elif [ "$MODE" = "cpu" ]; then
    ENV_HEADER="$SRC_DIR/$ENV.h"
    if ! grep -q 'typedef[[:space:]].*obs_t' "$ENV_HEADER" 2>/dev/null; then
        echo "Error: $ENV_HEADER must typedef obs_t for standalone eval"
        exit 1
    fi

    echo "Compiling standalone CPU eval for $ENV..."
    ${CC:-clang} "${CLANG_OPT[@]}" \
        -I. -Isrc -I$SRC_DIR -Ivendor "${INCLUDES[@]}" \
        -DPLATFORM_DESKTOP \
        -DPUFFERCPU_EVAL_MAIN \
        -DENV_HEADER=\"$ENV_HEADER\" \
        -x c src/puffercpu.h -x none $EXTRA_SRC \
        "${LINK_ARCHIVES[@]}" \
        "${EXTRA_LDFLAGS[@]}" \
        "${STANDALONE_LDFLAGS[@]}" \
        -lm -lpthread -fopenmp \
        -o build_cpu
    echo "Built: ./build_cpu"
    exit 0
fi

CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-$(dirname "$(dirname "$(which nvcc)")")}}
# NCCL include/lib fallback.
# Needed when NCCL is provided by the nvidia-nccl-cu12 wheel in the active venv.
NCCL_IFLAG=""
NCCL_LFLAG=""
for dir in /usr/include /usr/local/cuda/include; do
    if [ -f "$dir/nccl.h" ]; then NCCL_IFLAG="-I$dir"; break; fi
done
for dir in /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64; do
    if [ -f "$dir/libnccl.so" ] || [ -f "$dir/libnccl.so.2" ]; then NCCL_LFLAG="-L$dir"; break; fi
done
if [ -z "$NCCL_IFLAG" ]; then
    NCCL_IFLAG=$(python -c "import nvidia.nccl, os; print('-I' + os.path.join(nvidia.nccl.__path__[0], 'include'))" 2>/dev/null || echo "")
fi
if [ -z "$NCCL_LFLAG" ]; then
    NCCL_LFLAG=$(python -c "import nvidia.nccl, os; print('-L' + os.path.join(nvidia.nccl.__path__[0], 'lib'))" 2>/dev/null || echo "")
fi

export CCACHE_DIR="${CCACHE_DIR:-$HOME/.ccache}"
export CCACHE_BASEDIR="$(pwd)"
export CCACHE_COMPILERCHECK=content
NVCC="ccache $CUDA_HOME/bin/nvcc"
CC="${CC:-$(command -v ccache >/dev/null && echo 'ccache clang' || echo 'clang')}"
ARCH=${NVCC_ARCH:-native}

ENV_HEADER="$SRC_DIR/$ENV.h"
mkdir -p build
if ! grep -q 'typedef[[:space:]].*obs_t' "$ENV_HEADER" 2>/dev/null; then
    echo "Error: $ENV_HEADER must typedef obs_t"
    exit 1
fi

ENV_COMPILE_FLAGS=(-DENV_HEADER=\"$ENV_HEADER\")
GPU_ENV_HEADER="$SRC_DIR/$ENV.cu"
if [ -f "$GPU_ENV_HEADER" ]; then
    ENV_COMPILE_FLAGS+=(-DGPU_ENV_HEADER=\"$GPU_ENV_HEADER\")
fi

MODE=${MODE:-native}

if [ "$MODE" = "native" ]; then
    echo "Compiling native train/eval binary ($ARCH)..."
    $NVCC $NVCC_OPT -arch=$ARCH -std=c++17 \
        -I. -Isrc -I$SRC_DIR -Ivendor \
        -I$CUDA_HOME/include $NCCL_IFLAG -I$RAYLIB_NAME/include \
	    "${ENV_COMPILE_FLAGS[@]}" \
	    -DENV_NAME=$ENV \
	    -DPUFFERLIB_BUILD_MAIN \
	    -Xcompiler=-DPLATFORM_DESKTOP \
	    -Xcompiler=-fopenmp \
	    $PRECISION \
	    src/pufferl.cu \
        "$RAYLIB_A" \
        -L$CUDA_HOME/lib64 $NCCL_LFLAG \
        "${EXTRA_LDFLAGS[@]}" \
        -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
        -lm -lpthread $OMP_LIB "${STANDALONE_LDFLAGS[@]}" \
        -o puffer
    echo "Built: ./puffer"

elif [ "$MODE" = "profile" ]; then
    echo "Compiling profile binary ($ARCH)..."
    $NVCC $NVCC_OPT -arch=$ARCH -std=c++17 \
        -I. -Isrc -I$SRC_DIR -Ivendor \
        -I$CUDA_HOME/include $NCCL_IFLAG -I$RAYLIB_NAME/include \
        "${ENV_COMPILE_FLAGS[@]}" \
        -DENV_NAME=$ENV \
        -Xcompiler=-DPLATFORM_DESKTOP \
        $PRECISION \
        -Xcompiler=-fopenmp \
        tests/profile_kernels.cu \
        "$RAYLIB_A" \
        -lnccl -lnvidia-ml -lcublas -lcurand \
        -lGL -lm -lpthread $OMP_LIB \
        -o profile
    echo "Built: ./profile"
fi
