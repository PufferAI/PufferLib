#!/bin/bash
set -e

# Usage:
#   ./build.sh breakout              # Native train/eval -> ./puffer
#   ./build.sh breakout mybin        # Native -> ./mybin (does not clobber ./puffer)
#   ./build.sh breakout --cu         # CUDA env (ENV_HEADER=ocean/ENV/ENV.cu; exclusive vs .h)
#   ./build.sh robot_arm             # CUDA-only; implies --cu
#   ./build.sh breakout --float      # float32 precision (required for --slowly)
#   ./build.sh breakout --cpu        # Play/eval binary (optimized) -> ./ENV
#   ./build.sh osrs_inferno --cpu     # OSRS visual policy viewer -> ./osrs_inferno
#   ./build.sh breakout myplay --cpu # Play -> ./myplay
#   ./build.sh breakout --debug      # Debug (-O0 -g; sanitizers on --cpu)
#   ./build.sh breakout --web        # Emscripten web build
#                                    # copy build/web/ENV/* to ../docker/puffer.ai/docs/assets/ENV/
#   ./build.sh breakout --profile    # Kernel profiling binary
#   ./build.sh constellation         # Sweep dashboard -> ./seethestars
#   ./build.sh cache_data            # Sweep log cache -> ./cache_data
#   ./build.sh trailer               # 5.0 trailer -> ./trailer/trailer (also exports diagrams)
#   ./build.sh all                   # Build all envs native and native float32
#
# Env is compiled in. Run: ./puffer train|eval|match|sweep [--section.key=value ...]

if [ -z "$1" ]; then
    echo "Usage: ./build.sh ENV [OUT] [--cu] [--float] [--debug] [--cpu] [--web] [--profile]"
    exit 1
fi
ENV=$1
shift
OUT=""
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
    OUT=$1
    shift
fi

USE_GPU_ENV=0
SNAKE_RAW=0
while [ $# -gt 0 ]; do
    case $1 in
        --cu) USE_GPU_ENV=1 ;;
        --float) PRECISION="-DPRECISION_FLOAT" ;;
        --no-onehot) SNAKE_RAW=1 ;;
        --debug) DEBUG=1 ;;
        --web)   MODE=web ;;
        --profile) MODE=profile ;;
        --cpu)   MODE=cpu ;;
        *) echo "Error: unknown argument '$1'" && exit 1 ;;
    esac
    shift
done

if [ "$ENV" = "robot_arm" ]; then
    USE_GPU_ENV=1
    case "${MODE:-native}" in
        cpu|web)
            echo "Error: robot_arm physics is CUDA-only; use the native trainer build" >&2
            exit 1
            ;;
    esac
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

# Linux/mac
PLATFORM="$(uname -s)"
if [ "$PLATFORM" = "Linux" ]; then
    RAYLIB_NAME='raylib-5.5_linux_amd64'
    OMP_FLAGS=(-fopenmp)
    OMP_LIB=-lomp5
    SANITIZE_FLAGS=(-fsanitize=address,undefined,bounds,pointer-overflow,leak -fno-omit-frame-pointer)
    STANDALONE_LDFLAGS=(-lGL)
else
    RAYLIB_NAME='raylib-5.5_macos'
    OMP_PREFIX="$(brew --prefix libomp)"
    OMP_FLAGS=(-Xclang -fopenmp -I"$OMP_PREFIX/include" -L"$OMP_PREFIX/lib" -lomp)
    OMP_LIB=-lomp
    SANITIZE_FLAGS=()
    STANDALONE_LDFLAGS=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
fi

CLANG_WARN=(
    -Wall
    -Wno-narrowing
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
if [ -n "${NVCC_EXTRA:-}" ]; then
    read -ra _nvcc_extra <<< "$NVCC_EXTRA"
    EXTRA_CFLAGS+=("${_nvcc_extra[@]}")
fi
SRC_FILE=""

if [ "$ENV" = "clifford" ]; then
    EXTRA_CFLAGS+=(-DCLIFFORD_USE_SHORTCUT_GATES="${CLIFFORD_USE_SHORTCUT_GATES:-0}")
    EXTRA_CFLAGS+=(-DCLIFFORD_PAIR_ONEHOT="${CLIFFORD_PAIR_ONEHOT:-0}")
    if [ -n "${CLIFFORD_N_QUBITS:-}" ]; then
        EXTRA_CFLAGS+=(-DCLIFFORD_N_QUBITS="$CLIFFORD_N_QUBITS")
    fi
fi

if [ "$ENV" = "constellation" ]; then
    SRC_DIR="src"
    OUTPUT_NAME="seethestars"
    STANDALONE=1
    CLANG_WARN+=(-Wno-unused-function)
elif [ "$ENV" = "cache_data" ]; then
    SRC_DIR="src"
    OUTPUT_NAME="cache_data"
    SRC_FILE="src/constellation.c"
    EXTRA_CFLAGS+=(-DPUFFER_CACHE_DATA)
    STANDALONE=1
    CLANG_WARN+=(-Wno-unused-function)
elif [ "$ENV" = "trailer" ]; then
    SRC_DIR="trailer"
    OUTPUT_NAME="trailer/trailer"
    SRC_FILE="trailer/puffer5.c"
    EXTRA_SRC="trailer/architecture.c trailer/stars.c trailer/plot_scale.c"
    EXTRA_CFLAGS+=(-DARCHITECTURE_LIB)
    STANDALONE=1
    CLANG_WARN+=(-Wno-unused-function)
elif [ "$ENV" = "impulse_wars" ]; then
    SRC_DIR="ocean/$ENV"
    if [ "$MODE" = "web" ]; then BOX2D_NAME='box2d-web'
    elif [ "$PLATFORM" = "Linux" ]; then BOX2D_NAME='box2d-linux-amd64'
    else BOX2D_NAME='box2d-macos-arm64'
    fi
    BOX2D_URL="https://github.com/capnspacehook/box2d/releases/latest/download"
    download "$BOX2D_NAME" "$BOX2D_URL/$BOX2D_NAME.tar.gz"
    INCLUDES+=(-I./$BOX2D_NAME/include -I./$BOX2D_NAME/src -I./ocean/impulse_wars -I./vendor/collections-c)
    LINK_ARCHIVES+=("./$BOX2D_NAME/libbox2d.a")
    # C++ trainer only: game is C (void*/compound literals), not C++17.
    if [ -z "${MODE:-}" ] || [ "$MODE" = "native" ] || [ "$MODE" = "profile" ]; then
        EXTRA_SRC="ocean/impulse_wars/impulse_wars_api.c"
    fi
elif [ "$ENV" = "nethack" ]; then
    SRC_DIR="ocean/$ENV"
    EXTRA_CFLAGS+=(-DPUFFER_NETHACK)
    NLE_DIR="vendor/fast-nle"
    NLE_REPO="https://github.com/FinlaySanders/fast-nle.git"
    if [ ! -d "$NLE_DIR/src" ]; then
        echo "Cloning fast-nle from $NLE_REPO ..."
        git clone --depth 1 "$NLE_REPO" "$NLE_DIR"
    fi
    NETHACK_LIB_DIR="$(pwd)/$NLE_DIR/build"
    if [ ! -f "$NETHACK_LIB_DIR/libnethack.so" ]; then
        echo "Building libnethack.so ..."
        cmake -S "$NLE_DIR" -B "$NETHACK_LIB_DIR" -DCMAKE_BUILD_TYPE=Release
        cmake --build "$NETHACK_LIB_DIR" --target nethack -j$(nproc)
    fi
    INCLUDES+=(-I./$NLE_DIR/include
               -I./$NLE_DIR/build/_deps/deboost_context-src/include)
    EXTRA_LDFLAGS+=(-L"$NETHACK_LIB_DIR" -lnethack
                    -Xlinker -rpath -Xlinker "$NETHACK_LIB_DIR" -ldl)
elif [ -d "ocean/$ENV" ]; then
    SRC_DIR="ocean/$ENV"
else
    echo "Error: environment '$ENV' not found" && exit 1
fi

# src/ocean.cu compiles only this env's custom net (PUFFER_NETHACK, PUFFER_NMMO3, …).
EXTRA_CFLAGS+=(-DPUFFER_${ENV^^})

case "$ENV" in
    osrs_*)
        python3 ocean/osrs/scripts/osrs_asset_manifest.py generate-c-header \
            ocean/osrs/asset_manifest.json \
            --output ocean/osrs/osrs_assets_generated.h
        bash ocean/osrs/scripts/setup-data.sh
        ;;
esac

USER_OUTPUT_NAME=${OUTPUT_NAME-}
OUTPUT_NAME=${OUTPUT_NAME:-$ENV}
if [ -n "$OUT" ]; then
    OUTPUT_NAME=$OUT
fi
SRC_FILE=${SRC_FILE:-$SRC_DIR/$ENV.c}

if [ "$(uname -m)" = "x86_64" ]; then
    SIMD_FLAGS=(-mavx2 -mfma)
else
    SIMD_FLAGS=()
fi
if [ -n "$DEBUG" ] || [ "$MODE" = "local" ]; then
    CLANG_OPT=(-g -O0 "${CLANG_WARN[@]}" "${SANITIZE_FLAGS[@]}" "${SIMD_FLAGS[@]}")
    NVCC_OPT="-O0 -g"
    LINK_OPT="-g"
else
# No -DNDEBUG: keep assert() active (train/sweep fail-fast with messages).
    CLANG_OPT=(-O2 "${CLANG_WARN[@]}" "${SIMD_FLAGS[@]}")
    NVCC_OPT="-O2 --threads 0"
    LINK_OPT="-O2"
fi
# Dashboard / cache / trailer: compile SRC_FILE only (not puffercpu / CUDA / obs_t).
if [ "$STANDALONE" = "1" ]; then
    if [ "$MODE" = "web" ] || [ "$MODE" = "profile" ]; then
        echo "Error: $ENV is a standalone app, not an env" >&2
        exit 1
    fi
    echo "Compiling $ENV..."
    ${CC:-clang} "${CLANG_OPT[@]}" \
        -I. "${INCLUDES[@]}" \
        "$SRC_FILE" $EXTRA_SRC -o "$OUTPUT_NAME" \
        "${LINK_ARCHIVES[@]}" \
        "${EXTRA_LDFLAGS[@]}" \
        "${STANDALONE_LDFLAGS[@]}" \
        -lm -lpthread \
        -DPLATFORM_DESKTOP \
        "${EXTRA_CFLAGS[@]}"
    echo "Built: ./$OUTPUT_NAME"
    if [ "$ENV" = "trailer" ]; then
        echo "Compiling architecture..."
        ${CC:-clang} "${CLANG_OPT[@]}" \
            -I. "${INCLUDES[@]}" \
            trailer/architecture.c -o trailer/architecture \
            "${LINK_ARCHIVES[@]}" \
            "${EXTRA_LDFLAGS[@]}" \
            "${STANDALONE_LDFLAGS[@]}" \
            -lm -lpthread \
            -DPLATFORM_DESKTOP
        echo "Built: ./trailer/architecture"
        echo "Exporting diagrams..."
        mkdir -p trailer/shots
        ./trailer/architecture --shot trailer/shots/loop.png
        ./"$OUTPUT_NAME" --plot trailer/shots/fig_scale.png
    fi
    exit 0
fi
if [ "$MODE" = "cpu" ]; then
    STANDALONE_SOURCE="src/puffercpu.c"
    STANDALONE_DEFINES=()
    case "$ENV" in
        osrs_*)
            STANDALONE_SOURCE="$SRC_FILE"
            ;;
        *)
            ENV_HEADER="$SRC_DIR/$ENV.h"
            if ! grep -q 'typedef[[:space:]].*obs_t' "$ENV_HEADER" 2>/dev/null; then
                echo "Error: $ENV_HEADER must typedef obs_t for standalone eval"
                exit 1
            fi
            STANDALONE_DEFINES=(
                -DPUFFERCPU_EVAL_MAIN
                -DENV_HEADER=\"$ENV_HEADER\"
                -DPUFFER_ENV_NAME=\"$ENV\"
            )
            ;;
    esac
    FLAGS=(
        -I. -Isrc -I$SRC_DIR -Ivendor "${INCLUDES[@]}"
        "$STANDALONE_SOURCE" $EXTRA_SRC -o "$OUTPUT_NAME"
        "${LINK_ARCHIVES[@]}"
        "${EXTRA_LDFLAGS[@]}"
        "${STANDALONE_LDFLAGS[@]}"
        -lm -lpthread "${OMP_FLAGS[@]}"
        -DPLATFORM_DESKTOP
        "${STANDALONE_DEFINES[@]}"
        "${EXTRA_CFLAGS[@]}"
    )
    echo "Compiling $ENV..."
    ${CC:-clang} "${CLANG_OPT[@]}" "${FLAGS[@]}"
    echo "Built: ./$OUTPUT_NAME"
    exit 0
elif [ "$MODE" = "web" ]; then
    if [ -f "$SRC_DIR/web.sh" ]; then
        source "$SRC_DIR/web.sh"
        exit 0
    fi
    ENV_HEADER="$SRC_DIR/$ENV.h"
    if ! grep -q 'typedef[[:space:]].*obs_t' "$ENV_HEADER" 2>/dev/null; then
        echo "Error: $ENV_HEADER must typedef obs_t for web eval"
        exit 1
    fi
    mkdir -p "build/web/$ENV"
    PRELOAD_ENV=()
    if [ "$ENV" = "boxoban" ]; then
        # Do not pack generated boxoban_maps_*.bin or levels/ (hundreds of MB).
        PRELOAD_ENV=(
            --preload-file resources/boxoban/web_maps.bin@resources/boxoban/web_maps.bin
            --preload-file resources/boxoban/boxoban_weights.bin@resources/boxoban/boxoban_weights.bin
            --preload-file resources/boxoban/Wall_Black.jpg@resources/boxoban/Wall_Black.jpg
            --preload-file resources/boxoban/Crate_Black.jpg@resources/boxoban/Crate_Black.jpg
            --preload-file resources/boxoban/EndPoint_Black.jpg@resources/boxoban/EndPoint_Black.jpg
            --preload-file resources/boxoban/EndPoint_Blue.jpg@resources/boxoban/EndPoint_Blue.jpg
            --preload-file resources/boxoban/GroundGravel_Concrete.jpg@resources/boxoban/GroundGravel_Concrete.jpg
        )
    elif [ -d "resources/$ENV" ]; then
        PRELOAD_ENV=(--preload-file "resources/$ENV@resources/$ENV")
    fi
    echo "Compiling $ENV for web..."
    PRELOAD=(
        --preload-file resources/shared@resources/shared
        --preload-file config/default.ini@config/default.ini
    )
    if [ -d "ocean/$ENV/generated" ]; then
        PRELOAD+=(--preload-file "ocean/$ENV/generated@ocean/$ENV/generated")
    fi
    if [ -f "config/$ENV.ini" ]; then
        PRELOAD+=(--preload-file "config/$ENV.ini@config/$ENV.ini")
    fi
    # Web overlays live on the site, not in this repo.
    SITE_WEB_INI="../docker/puffer.ai/config/${ENV}_web.ini"
    if [ -f "$SITE_WEB_INI" ]; then
        PRELOAD+=(--preload-file "$SITE_WEB_INI@config/${ENV}_web.ini")
    elif [ -f "config/${ENV}_web.ini" ]; then
        PRELOAD+=(--preload-file "config/${ENV}_web.ini@config/${ENV}_web.ini")
    fi
    emcc \
        -o "build/web/$ENV/game.html" \
        src/puffercpu.c $EXTRA_SRC \
        -O3 -Wall -Wno-narrowing \
        "${LINK_ARCHIVES[@]}" \
        -I. -Isrc -I$SRC_DIR -Ivendor "${INCLUDES[@]}" \
        -L. -L./$RAYLIB_NAME/lib \
        -sASSERTIONS=2 -gsource-map \
        -sUSE_GLFW=3 -sUSE_WEBGL2=1 -sASYNCIFY -sFILESYSTEM -sFORCE_FILESYSTEM=1 \
        --js-library vendor/puf_web_vsync.js \
        --shell-file vendor/minshell.html \
        -sINITIAL_MEMORY=512MB -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=512KB \
        -DPLATFORM_WEB -DGRAPHICS_API_OPENGL_ES3 \
        -DPUFFERCPU_EVAL_MAIN \
        -DENV_HEADER=\"$ENV_HEADER\" \
        -DPUFFER_ENV_NAME=\"$ENV\" \
        --preload-file resources/shared@resources/shared \
        "${PRELOAD_ENV[@]}" \
        "${PRELOAD[@]}" \
        "${EXTRA_CFLAGS[@]}"
    echo "Built: build/web/$ENV/game.html"
    WEBSITE_DIR="${PUFFER_WEBSITE_DIR:-../docker/puffer.ai}"
    WEBSITE_ASSETS="$WEBSITE_DIR/docs/assets"
    if [ -d "$WEBSITE_ASSETS" ]; then
        mkdir -p "$WEBSITE_ASSETS/$ENV"
        cp -a "build/web/$ENV/." "$WEBSITE_ASSETS/$ENV/"
        echo "Published: $WEBSITE_ASSETS/$ENV/"
    fi
    exit 0
elif [ "$MODE" = "cpu" ]; then
    ENV_HEADER="$SRC_DIR/$ENV.h"
    if ! grep -q 'typedef[[:space:]].*obs_t' "$ENV_HEADER" 2>/dev/null; then
        echo "Error: $ENV_HEADER must typedef obs_t for standalone eval"
        exit 1
    fi

    mkdir -p build
    echo "Compiling standalone CPU eval for $ENV..."
    ${CC:-clang} "${CLANG_OPT[@]}" \
        -I. -Isrc -I$SRC_DIR -Ivendor "${INCLUDES[@]}" \
        -DPLATFORM_DESKTOP \
        -DPUFFERCPU_EVAL_MAIN \
        -DENV_HEADER=\"$ENV_HEADER\" \
        -DPUFFER_ENV_NAME=\"$ENV\" \
        -x c src/puffercpu.h -x none $EXTRA_SRC \
        "${LINK_ARCHIVES[@]}" \
        "${EXTRA_LDFLAGS[@]}" \
        "${STANDALONE_LDFLAGS[@]}" \
        -lm -lpthread "${OMP_FLAGS[@]}" \
        -o "build/cpu_${ENV}"
    echo "Built: ./build/cpu_${ENV}"
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

# CPU and CUDA envs are separate sources. --cu selects the .cu; default is .h.
# Only one is compiled in (never both).
if [ "$USE_GPU_ENV" = "1" ]; then
    ENV_HEADER="$SRC_DIR/$ENV.cu"
    if [ ! -f "$ENV_HEADER" ]; then
        echo "Error: --cu requires $ENV_HEADER"
        exit 1
    fi
else
    ENV_HEADER="$SRC_DIR/$ENV.h"
fi
mkdir -p build
if ! grep -q 'typedef[[:space:]].*obs_t' "$ENV_HEADER" 2>/dev/null; then
    echo "Error: $ENV_HEADER must typedef obs_t"
    exit 1
fi

ENV_COMPILE_FLAGS=(-DENV_HEADER=\"$ENV_HEADER\")

MODE=${MODE:-native}

# Allow double→int/float in brace-init (host -Wno-narrowing + nvcc #2361).
NVCC_NARROW=(-Xcompiler=-Wno-narrowing --diag-suppress=2361)

if [ "$MODE" = "native" ]; then
    if [ -n "$OUT" ]; then
        TRAIN_BIN="$OUT"
    elif [ -n "$USER_OUTPUT_NAME" ]; then
        TRAIN_BIN="$USER_OUTPUT_NAME"
    else
        TRAIN_BIN="puffer"
    fi
    if [ "$SNAKE_RAW" = "1" ]; then
        EXTRA_CFLAGS+=(-DSNAKE_ONEHOT=0)
    fi
    OSRS_RENDER_OBJECT=""
    case "$ENV" in
        osrs_*)
            OSRS_RENDER_OBJECT="build/osrs_puffer_render.o"
            ENV_COMPILE_FLAGS+=(-DOSRS_PUFFER_RENDER)
            $CC $LINK_OPT "${CLANG_WARN[@]}" "${SIMD_FLAGS[@]}" -std=c11 \
                -I. -Isrc -I$SRC_DIR -Ivendor \
                "${INCLUDES[@]}" \
                -DPLATFORM_DESKTOP \
                -c ocean/osrs/osrs_puffer_render.c \
                -o "$OSRS_RENDER_OBJECT"
            ;;
    esac
    echo "Compiling native train/eval binary ($ARCH) -> $TRAIN_BIN..."
    $NVCC $NVCC_OPT -arch=$ARCH -std=c++17 \
        -I. -Isrc -I$SRC_DIR -Ivendor \
        "${INCLUDES[@]}" \
        -I$CUDA_HOME/include -I$CUDA_HOME/include/cccl $NCCL_IFLAG -I$RAYLIB_NAME/include \
	    "${ENV_COMPILE_FLAGS[@]}" \
	    -DENV_NAME=$ENV \
	    -DPUFFER_ENV_NAME=\"$ENV\" \
	    -DPUFFERLIB_BUILD_MAIN \
	    -Xcompiler=-DPLATFORM_DESKTOP \
	    -Xcompiler=-fopenmp \
	    "${NVCC_NARROW[@]}" \
	    "${EXTRA_CFLAGS[@]}" \
	    $PRECISION \
	    src/pufferl.cu \
        $EXTRA_SRC \
        $OSRS_RENDER_OBJECT \
        "${LINK_ARCHIVES[@]}" \
        -L$CUDA_HOME/lib64 $NCCL_LFLAG \
        "${EXTRA_LDFLAGS[@]}" \
        -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
        -lm -lpthread $OMP_LIB "${STANDALONE_LDFLAGS[@]}" \
        -o "$TRAIN_BIN"
    echo "Built: ./$TRAIN_BIN"

elif [ "$MODE" = "profile" ]; then
    PROFILE_BIN="build/profile_${ENV}"
    echo "Compiling profile binary ($ARCH) -> $PROFILE_BIN..."
    $NVCC $NVCC_OPT -arch=$ARCH -std=c++17 \
        -I. -Isrc -I$SRC_DIR -Ivendor \
        "${INCLUDES[@]}" \
        -I$CUDA_HOME/include -I$CUDA_HOME/include/cccl $NCCL_IFLAG -I$RAYLIB_NAME/include \
        "${ENV_COMPILE_FLAGS[@]}" \
        -DENV_NAME=$ENV \
	    -DPUFFER_ENV_NAME=\"$ENV\" \
        -Xcompiler=-DPLATFORM_DESKTOP \
	    "${NVCC_NARROW[@]}" \
	    "${EXTRA_CFLAGS[@]}" \
        $PRECISION \
        -Xcompiler=-fopenmp \
        tests/profile_kernels.cu \
        "$RAYLIB_A" \
        -L$CUDA_HOME/lib64 \
        -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
        -lGL -lm -lpthread $OMP_LIB \
        -o "$PROFILE_BIN"
    echo "Built: ./$PROFILE_BIN"
fi
