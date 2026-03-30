#!/bin/bash
set -e

# Usage:
#   ./build.sh breakout              # Build _C.so with breakout statically linked
#   ./build.sh breakout --float      # float32 precision (required for --slowly)
#   ./build.sh breakout --debug      # Debug build
#   ./build.sh breakout --local      # Standalone executable (debug, sanitizers)
#   ./build.sh breakout --fast       # Standalone executable (optimized)
#   ./build.sh breakout --web        # Emscripten web build
#   ./build.sh breakout --profile    # Kernel profiling binary

ENV=${1:?Usage: ./build.sh ENV_NAME [--float] [--debug] [--local|--fast|--web|--profile|--cpu]}
MODE=""
PRECISION=""
DEBUG=""
for arg in "${@:2}"; do
    case $arg in
        --float) PRECISION="-DPRECISION_FLOAT" ;;
        --debug) DEBUG=1 ;;
        --local) MODE=local ;;
        --fast)  MODE=fast ;;
        --web)   MODE=web ;;
        --profile) MODE=profile ;;
        --cpu)   MODE=cpu; PRECISION="-DPRECISION_FLOAT" ;;
    esac
done

CLANG_WARN="\
    -Wall \
    -ferror-limit=3 \
    -Werror=incompatible-pointer-types \
    -Werror=return-type \
    -Wno-error=incompatible-pointer-types-discards-qualifiers \
    -Wno-incompatible-pointer-types-discards-qualifiers \
    -Wno-error=array-parameter"

PLATFORM="$(uname -s)"

if [ -n "$DEBUG" ] || [ "$MODE" = "local" ]; then
    CLANG_OPT="-g -O0 $CLANG_WARN"
    NVCC_OPT="-O0 -g"
    LINK_OPT="-g"
    [ "$PLATFORM" = "Linux" ] && CLANG_OPT="$CLANG_OPT -fsanitize=address,undefined,bounds,pointer-overflow,leak -fno-omit-frame-pointer"
else
    CLANG_OPT="-O2 -DNDEBUG $CLANG_WARN"
    NVCC_OPT="-O3"
    LINK_OPT="-O2"
fi

# ============================================================================
# Platform + dependencies
# ============================================================================
if [ -d "ocean/$ENV" ]; then
    SRC_DIR="ocean/$ENV"
elif [ -d "$ENV" ]; then
    SRC_DIR="$ENV"
else
    echo "Error: environment '$ENV' not found" && exit 1
fi

if [ "$PLATFORM" = "Linux" ]; then
    RAYLIB_NAME='raylib-5.5_linux_amd64'
else
    RAYLIB_NAME='raylib-5.5_macos'
fi

RAYLIB_URL="https://github.com/raysan5/raylib/releases/download/5.5"

download() {
    local name=$1 url=$2
    [ -d "$name" ] && return
    echo "Downloading $name..."
    if [[ "$url" == *.zip ]]; then
        curl -sL "$url" -o "$name.zip" && unzip -q "$name.zip" && rm "$name.zip"
    else
        curl -sL "$url" -o "$name.tar.gz" && tar xf "$name.tar.gz" && rm "$name.tar.gz"
    fi
}

# Raylib (always needed)
if [ "$MODE" = "web" ]; then
    RAYLIB_NAME='raylib-5.5_webassembly'
    download "$RAYLIB_NAME" "$RAYLIB_URL/$RAYLIB_NAME.zip"
else
    download "$RAYLIB_NAME" "$RAYLIB_URL/$RAYLIB_NAME.tar.gz"
fi
[ ! -f "$RAYLIB_NAME/include/rlights.h" ] && \
    curl -sL "https://raw.githubusercontent.com/raysan5/raylib/master/examples/shaders/rlights.h" \
        -o "$RAYLIB_NAME/include/rlights.h"

RAYLIB_A="$RAYLIB_NAME/lib/libraylib.a"
INCLUDES=(-I./$RAYLIB_NAME/include -I./src)
LINK_ARCHIVES="$RAYLIB_A"
EXTRA_SRC=""

# Box2d (impulse_wars only)
if [ "$ENV" = "impulse_wars" ]; then
    if [ "$MODE" = "web" ]; then BOX2D_NAME='box2d-web'
    elif [ "$PLATFORM" = "Linux" ]; then BOX2D_NAME='box2d-linux-amd64'
    else BOX2D_NAME='box2d-macos-arm64'
    fi
    BOX2D_URL="https://github.com/capnspacehook/box2d/releases/latest/download"
    download "$BOX2D_NAME" "$BOX2D_URL/$BOX2D_NAME.tar.gz"
    INCLUDES+=(-I./$BOX2D_NAME/include -I./$BOX2D_NAME/src)
    LINK_ARCHIVES="$LINK_ARCHIVES ./$BOX2D_NAME/libbox2d.a"
fi

# Constellation needs cJSON
[ "$ENV" = "constellation" ] && EXTRA_SRC="vendor/cJSON.c" && INCLUDES+=(-I./vendor) && OUTPUT_NAME="seethestars"

# ============================================================================
# Standalone builds: --local, --fast, --web
# ============================================================================

if [ "$MODE" = "web" ]; then
    [ ! -f "minshell.html" ] && \
        curl -sL "https://raw.githubusercontent.com/raysan5/raylib/master/src/minshell.html" -o minshell.html
    mkdir -p "build_web/$ENV"
    echo "Building $ENV for web..."
    emcc \
        -o "build_web/$ENV/game.html" \
        "$SRC_DIR/$ENV.c" $EXTRA_SRC \
        -O3 -Wall \
        $LINK_ARCHIVES \
        "${INCLUDES[@]}" \
        -L. -L./$RAYLIB_NAME/lib \
        -sASSERTIONS=2 -gsource-map \
        -sUSE_GLFW=3 -sUSE_WEBGL2=1 -sASYNCIFY -sFILESYSTEM -sFORCE_FILESYSTEM=1 \
        --shell-file ./minshell.html \
        -sINITIAL_MEMORY=512MB -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=512KB \
        -DNDEBUG -DPLATFORM_WEB -DGRAPHICS_API_OPENGL_ES3 \
        --preload-file resources/$ENV@resources/$ENV \
        --preload-file resources/shared@resources/shared
    echo "Built: build_web/$ENV/game.html"
    exit 0
fi

if [ "$MODE" = "local" ] || [ "$MODE" = "fast" ]; then
    FLAGS=(
        "${INCLUDES[@]}"
        "$SRC_DIR/$ENV.c" $EXTRA_SRC -o "${OUTPUT_NAME:-$ENV}"
        $LINK_ARCHIVES
        -lm -lpthread -fopenmp
        -DPLATFORM_DESKTOP
    )
    [ "$PLATFORM" = "Darwin" ] && FLAGS+=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
    ${CC:-clang} $CLANG_OPT "${FLAGS[@]}"
    echo "Built: ./${OUTPUT_NAME:-$ENV}"
    exit 0
fi

# ============================================================================
# Default: build _C.so with env statically linked
# ============================================================================

CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}
NVCC="$CUDA_HOME/bin/nvcc"

PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INCLUDE=$(python -c "import pybind11; print(pybind11.get_include())")
NUMPY_INCLUDE=$(python -c "import numpy; print(numpy.get_include())")
EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTPUT="pufferlib/_C${EXT_SUFFIX}"

# Step 1: Static env library
BINDING_SRC="$SRC_DIR/binding.c"
STATIC_OBJ="src/libstatic_${ENV}.o"
STATIC_LIB="src/libstatic_${ENV}.a"
[ ! -f "$BINDING_SRC" ] && echo "Error: $BINDING_SRC not found" && exit 1

echo "=== Building static env: $ENV ==="
${CC:-clang} -c $CLANG_OPT \
    -I. -Isrc -I$SRC_DIR \
    -I./$RAYLIB_NAME/include -I$CUDA_HOME/include \
    -DPLATFORM_DESKTOP \
    -fno-semantic-interposition -fvisibility=hidden \
    -fPIC -fopenmp \
    "$BINDING_SRC" -o "$STATIC_OBJ"
ar rcs "$STATIC_LIB" "$STATIC_OBJ"

OBS_TENSOR_T=$(strings "$STATIC_OBJ" | grep 'Tensor$' | head -1)
[ -z "$OBS_TENSOR_T" ] && echo "Error: Could not find OBS_TENSOR_T" && exit 1
echo "OBS_TENSOR_T=$OBS_TENSOR_T"

# Step 2: Profile binary or Python bindings
if [ "$MODE" = "profile" ]; then
    ARCH=${NVCC_ARCH:-sm_89}
    echo "=== Building profile binary (arch=$ARCH) ==="
    $NVCC $NVCC_OPT -arch=$ARCH -std=c++17 \
        -I. -Isrc -I$SRC_DIR \
        -I$CUDA_HOME/include -I$RAYLIB_NAME/include \
        -DOBS_TENSOR_T=$OBS_TENSOR_T \
        -DENV_NAME=$ENV \
        -Xcompiler=-DPLATFORM_DESKTOP \
        $PRECISION \
        -Xcompiler=-fopenmp \
        tests/profile_kernels.cu ini.c \
        "$STATIC_LIB" "$RAYLIB_A" \
        -lnccl -lnvidia-ml -lcublas -lcurand -lcudnn -lnvToolsExt \
        -lGL -lm -lpthread -lomp \
        -o profile
    echo "=== Built: ./profile ==="
    exit 0
fi

if [ "$MODE" = "cpu" ]; then
    echo "=== Compiling bindings_cpu.cpp ==="
    ${CXX:-g++} -c -fPIC -fopenmp \
        -D_GLIBCXX_USE_CXX11_ABI=1 \
        -DPLATFORM_DESKTOP \
        -std=c++17 \
        -I. -Isrc \
        -I$PYTHON_INCLUDE -I$PYBIND_INCLUDE \
        -DOBS_TENSOR_T=$OBS_TENSOR_T \
        $PRECISION $LINK_OPT \
        src/bindings_cpu.cpp -o src/bindings_cpu.o

    echo "=== Linking $OUTPUT (CPU) ==="
    LINK_CMD=(
        ${CXX:-g++} -shared -fPIC -fopenmp
        src/bindings_cpu.o "$STATIC_LIB" "$RAYLIB_A"
        -lm -lpthread -lomp -undefined dynamic_lookup
        $LINK_OPT
    )
    [ "$PLATFORM" = "Linux" ] && LINK_CMD+=(-Bsymbolic-functions)
    [ "$PLATFORM" = "Darwin" ] && LINK_CMD+=(-framework Cocoa -framework OpenGL -framework IOKit)
    LINK_CMD+=(-o "$OUTPUT")
    "${LINK_CMD[@]}"
    echo "=== Built: $OUTPUT (CPU) ==="
    exit 0
fi

echo "=== Compiling bindings.cu ==="
$NVCC -c -Xcompiler -fPIC \
    -Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=1 \
    -Xcompiler=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
    -Xcompiler=-DPLATFORM_DESKTOP \
    -std=c++17 \
    -I. -Isrc \
    -I$PYTHON_INCLUDE -I$PYBIND_INCLUDE -I$NUMPY_INCLUDE \
    -I$CUDA_HOME/include -I$RAYLIB_NAME/include \
    -Xcompiler=-fopenmp \
    -DOBS_TENSOR_T=$OBS_TENSOR_T \
    $PRECISION $NVCC_OPT \
    src/bindings.cu -o src/bindings.o

# Step 3: Link
echo "=== Linking $OUTPUT ==="
LINK_CMD=(
    ${CXX:-g++} -shared -fPIC -fopenmp
    src/bindings.o "$STATIC_LIB" "$RAYLIB_A"
    -L$CUDA_HOME/lib64
    -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand -lcudnn
    -lnvToolsExt -lomp
    $LINK_OPT
)
[ "$PLATFORM" = "Linux" ] && LINK_CMD+=(-Bsymbolic-functions)
[ "$PLATFORM" = "Darwin" ] && LINK_CMD+=(-framework Cocoa -framework OpenGL -framework IOKit)
LINK_CMD+=(-o "$OUTPUT")
"${LINK_CMD[@]}"

echo "=== Built: $OUTPUT ==="
