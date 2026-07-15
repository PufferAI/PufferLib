#!/bin/bash
set -e

# Usage:
#   ./build.sh breakout              # Build _C.so with breakout statically linked
#   ./build.sh breakout --float      # float32 precision (required for --slowly)
#   ./build.sh breakout --cpu        # CPU fallback, torch only
#   ./build.sh breakout --debug      # Debug build
#   ./build.sh breakout --local      # Standalone executable (debug, sanitizers)
#   ./build.sh breakout --fast       # Standalone executable (optimized)
#   ./build.sh breakout --web        # Emscripten web build
#   ./build.sh breakout --profile    # Kernel profiling binary
#   ./build.sh all                   # Build all envs with default and --float

if [ -z "$1" ]; then
    echo "Usage: ./build.sh ENV_NAME [--float] [--debug] [--local|--fast|--web|--profile|--cpu|--all]"
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
        --cpu)   MODE=cpu; PRECISION="-DPRECISION_FLOAT" ;;
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
    SHARED_LDFLAGS=(-Bsymbolic-functions)
else
    RAYLIB_NAME='raylib-5.5_macos'
    OMP_LIB=-lomp
    SANITIZE_FLAGS=()
    STANDALONE_LDFLAGS=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
    SHARED_LDFLAGS=(-framework Cocoa -framework OpenGL -framework IOKit -undefined dynamic_lookup)
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

if [ "$ENV" = "constellation" ]; then
    SRC_DIR="constellation"
    EXTRA_SRC="vendor/cJSON.c"
    OUTPUT_NAME="seethestars"
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

# Standalone environment build
# -mavx2 enables AVX2 intrinsics (__m256, _mm256_*) which drive.h and
# src/bf16.h use directly. x86_64 only — strip if porting to ARM/Apple Silicon.
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
        "$SRC_DIR/$ENV.c" $EXTRA_SRC -o "$OUTPUT_NAME"
        "${LINK_ARCHIVES[@]}"
        "${EXTRA_LDFLAGS[@]}"
        "${STANDALONE_LDFLAGS[@]}"
        -lm -lpthread -fopenmp
        -DPLATFORM_DESKTOP
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
        "$SRC_DIR/$ENV.c" $EXTRA_SRC \
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
        --preload-file resources/shared@resources/shared
    echo "Built: build/web/$ENV/game.html"
    exit 0
fi

# Find cuDNN path
CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-$(dirname "$(dirname "$(which nvcc)")")}}
CUDNN_IFLAG=""
CUDNN_LFLAG=""
for dir in /usr/local/cuda/include /usr/include; do
    if [ -f "$dir/cudnn.h" ]; then
        CUDNN_IFLAG="-I$dir"
        break
    fi
done
for dir in /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu; do
    if [ -f "$dir/libcudnn.so" ]; then
        CUDNN_LFLAG="-L$dir"
        break
    fi
done
if [ -z "$CUDNN_IFLAG" ]; then
    CUDNN_IFLAG=$(python -c "import nvidia.cudnn, os; print('-I' + os.path.join(nvidia.cudnn.__path__[0], 'include'))" 2>/dev/null || echo "")
fi
if [ -z "$CUDNN_LFLAG" ]; then
    CUDNN_LFLAG=$(python -c "import nvidia.cudnn, os; print('-L' + os.path.join(nvidia.cudnn.__path__[0], 'lib'))" 2>/dev/null || echo "")
fi

# NCCL include/lib fallback (mirrors the cuDNN fallback above).
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

WHEEL_RPATH_FLAGS=()
for lib_flag in "$CUDNN_LFLAG" "$NCCL_LFLAG"; do
    if [[ "$lib_flag" == -L* ]]; then
        WHEEL_RPATH_FLAGS+=("-Wl,-rpath,${lib_flag#-L}")
    fi
done

export CCACHE_DIR="${CCACHE_DIR:-$HOME/.ccache}"
export CCACHE_BASEDIR="$(pwd)"
export CCACHE_COMPILERCHECK=content
NVCC="ccache $CUDA_HOME/bin/nvcc"
CC="${CC:-$(command -v ccache >/dev/null && echo 'ccache clang' || echo 'clang')}"
ARCH=${NVCC_ARCH:-native}

PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INCLUDE=$(python -c "import pybind11; print(pybind11.get_include())")
NUMPY_INCLUDE=$(python -c "import numpy; print(numpy.get_include())")
EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTPUT="pufferlib/_C${EXT_SUFFIX}"

BINDING_SRC="$SRC_DIR/binding.c"
mkdir -p build
STATIC_OBJ="build/libstatic_${ENV}.o"
STATIC_LIB="build/libstatic_${ENV}.a"

if [ ! -f "$BINDING_SRC" ]; then
    echo "Error: $BINDING_SRC not found"
    exit 1
fi

echo "Compiling static library for $ENV..."
${CC:-clang} -c "${CLANG_OPT[@]}" $EXTRA_CFLAGS \
    -I. -Isrc -I$SRC_DIR -Ivendor \
    "${INCLUDES[@]}" \
    -I./$RAYLIB_NAME/include -I$CUDA_HOME/include \
    -DPLATFORM_DESKTOP \
    -fno-semantic-interposition -fvisibility=hidden \
    -fPIC -fopenmp \
    "$BINDING_SRC" -o "$STATIC_OBJ"
ar rcs "$STATIC_LIB" "$STATIC_OBJ"

# Brittle hack: have to extract the tensor type from the static lib to build trainer
OBS_TENSOR_T=$(awk '/^#define OBS_TENSOR_T/{print $3}' "$BINDING_SRC")
if [ -z "$OBS_TENSOR_T" ]; then
    echo "Error: Could not find OBS_TENSOR_T in $BINDING_SRC"
    exit 1
fi

if [ -z "$MODE" ]; then
    echo "Compiling CUDA ($ARCH) training backend..."
    $NVCC -c -arch=$ARCH -Xcompiler -fPIC \
        -Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=1 \
        -Xcompiler=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
        -Xcompiler=-DPLATFORM_DESKTOP \
        -std=c++17 \
        -I. -Isrc \
        -I$PYTHON_INCLUDE -I$PYBIND_INCLUDE -I$NUMPY_INCLUDE \
        -I$CUDA_HOME/include $CUDNN_IFLAG $NCCL_IFLAG -I$RAYLIB_NAME/include \
        -Xcompiler=-fopenmp \
        -DOBS_TENSOR_T=$OBS_TENSOR_T \
        -DENV_NAME=$ENV \
        $PRECISION $NVCC_OPT \
        src/bindings.cu -o build/bindings.o

    LINK_CMD=(
        ${CXX:-g++} -shared -fPIC -fopenmp
        build/bindings.o "$STATIC_LIB" "$RAYLIB_A"
        -L$CUDA_HOME/lib64 $CUDNN_LFLAG $NCCL_LFLAG
        "${WHEEL_RPATH_FLAGS[@]}"
        "${EXTRA_LDFLAGS[@]}"
        -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand -lcudnn
        $OMP_LIB $LINK_OPT
        "${SHARED_LDFLAGS[@]}"
        -o "$OUTPUT"
    )
    "${LINK_CMD[@]}"
    echo "Built: $OUTPUT"

elif [ "$MODE" = "cpu" ]; then
    echo "Compiling CPU training backend..."
    ${CXX:-g++} -c -fPIC -fopenmp \
        -D_GLIBCXX_USE_CXX11_ABI=1 \
        -DPLATFORM_DESKTOP \
        -std=c++17 \
        -I. -Isrc \
        -I$PYTHON_INCLUDE -I$PYBIND_INCLUDE \
        -DOBS_TENSOR_T=$OBS_TENSOR_T \
        -DENV_NAME=$ENV \
        $PRECISION $LINK_OPT \
        src/bindings_cpu.cpp -o build/bindings_cpu.o
    LINK_CMD=(
        ${CXX:-g++} -shared -fPIC -fopenmp
        build/bindings_cpu.o "$STATIC_LIB" "$RAYLIB_A"
        "${EXTRA_LDFLAGS[@]}"
        -lm -lpthread $OMP_LIB $LINK_OPT
        "${SHARED_LDFLAGS[@]}"
        -o "$OUTPUT"
    )
    "${LINK_CMD[@]}"
    echo "Built: $OUTPUT"

elif [ "$MODE" = "profile" ]; then
    echo "Compiling profile binary ($ARCH)..."
    $NVCC $NVCC_OPT -arch=$ARCH -std=c++17 \
        -I. -Isrc -I$SRC_DIR -Ivendor \
        -I$CUDA_HOME/include $CUDNN_IFLAG $NCCL_IFLAG -I$RAYLIB_NAME/include \
        -DOBS_TENSOR_T=$OBS_TENSOR_T \
        -DENV_NAME=$ENV \
        -Xcompiler=-DPLATFORM_DESKTOP \
        $PRECISION \
        -Xcompiler=-fopenmp \
        tests/profile_kernels.cu vendor/ini.c \
        "$STATIC_LIB" "$RAYLIB_A" \
        -lnccl -lnvidia-ml -lcublas -lcurand -lcudnn \
        -lGL -lm -lpthread $OMP_LIB \
        -o profile
    echo "Built: ./profile"
fi
