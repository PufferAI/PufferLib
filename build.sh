#!/bin/bash
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: ./build.sh ENV_NAME [--float] [--debug] [--local|--fast|--web|--profile|--cpu|--all]"
    exit 1
fi

ENV=$1
shift

MODE=""
PRECISION=""
DEBUG=""
for arg in "$@"; do
    case $arg in
        --float) PRECISION="-DPRECISION_FLOAT" ;;
        --debug) DEBUG=1 ;;
        --local) MODE=local ;;
        --fast) MODE=fast ;;
        --web) MODE=web ;;
        --profile) MODE=profile ;;
        --cpu) MODE=cpu; PRECISION="-DPRECISION_FLOAT" ;;
        *) echo "Error: unknown argument '$arg'" && exit 1 ;;
    esac
done

if [ "$ENV" = "all" ]; then
    FAILED=""
    for env_dir in ocean/*/; do
        env_name=$(basename "$env_dir")
        if bash "$0" "$env_name" && bash "$0" "$env_name" --float; then
            echo "OK: $env_name"
        else
            echo "FAIL: $env_name"
            FAILED="$FAILED\n  $env_name"
        fi
    done
    if [ -n "$FAILED" ]; then
        echo -e "\nFailed builds:$FAILED"
        exit 1
    fi
    exit 0
fi

PLATFORM="$(uname -s)"

CLANG_WARN=(
    -Wall
    -ferror-limit=3
    -Werror=incompatible-pointer-types
    -Werror=return-type
    -Wno-error=incompatible-pointer-types-discards-qualifiers
    -Wno-incompatible-pointer-types-discards-qualifiers
    -Wno-error=array-parameter
)

if [ -n "$DEBUG" ] || [ "$MODE" = "local" ]; then
    CLANG_OPT=(-g -O0 "${CLANG_WARN[@]}")
    NVCC_OPT=(-O0 -g)
    LINK_OPT=(-g)
    if [ "$PLATFORM" = "Linux" ]; then
        CLANG_OPT+=(-fsanitize=address,undefined,bounds,pointer-overflow,leak)
        CLANG_OPT+=(-fno-omit-frame-pointer)
    fi
else
    CLANG_OPT=(-O2 -DNDEBUG "${CLANG_WARN[@]}")
    NVCC_OPT=(-O2 --threads 0)
    LINK_OPT=(-O2)
fi

download() {
    local name=$1
    local url=$2
    [ -d "$name" ] && return
    echo "Downloading $name..."
    case "$url" in
        *.zip) curl -sL "$url" -o "$name.zip" && unzip -q "$name.zip" && rm "$name.zip" ;;
        *) curl -sL "$url" -o "$name.tar.gz" && tar xf "$name.tar.gz" && rm "$name.tar.gz" ;;
    esac
}

find_omp_include() {
    if ! command -v brew >/dev/null 2>&1; then
        return 0
    fi
    local omp_prefix
    omp_prefix=$(brew --prefix libomp 2>/dev/null || true)
    if [ -n "$omp_prefix" ] && [ -d "$omp_prefix/include" ]; then
        echo "$omp_prefix/include"
    fi
}

DARWIN_OMP_SOURCE="system"
DARWIN_OMP_LINK=(-lomp)
resolve_darwin_omp_link() {
    local torch_omp=""
    torch_omp=$(python -c "import torch; import os; print(os.path.join(torch.__path__[0], 'lib', 'libomp.dylib'))" 2>/dev/null || true)
    if [ -n "$torch_omp" ] && [ -f "$torch_omp" ]; then
        local omp_dir
        omp_dir=$(dirname "$torch_omp")
        DARWIN_OMP_SOURCE="torch"
        DARWIN_OMP_LINK=(-L"$omp_dir" -Wl,-rpath,"$omp_dir" -lomp)
        return
    fi

    if command -v brew >/dev/null 2>&1; then
        local omp_prefix
        omp_prefix=$(brew --prefix libomp 2>/dev/null || true)
        if [ -n "$omp_prefix" ] && [ -d "$omp_prefix/lib" ]; then
            DARWIN_OMP_SOURCE="homebrew"
            DARWIN_OMP_LINK=(-L"$omp_prefix/lib" -Wl,-rpath,"$omp_prefix/lib" -lomp)
            return
        fi
    fi

    DARWIN_OMP_SOURCE="system"
    DARWIN_OMP_LINK=(-lomp)
}

if [ "$PLATFORM" = "Linux" ]; then
    RAYLIB_NAME='raylib-5.5_linux_amd64'
    OMP_LIB=-lomp5
    STANDALONE_LDFLAGS=(-lGL)
    SHARED_LDFLAGS=(-Bsymbolic-functions)
    DEFAULT_CC=${CC:-clang}
    DEFAULT_CXX=${CXX:-g++}
else
    RAYLIB_NAME='raylib-5.5_macos'
    STANDALONE_LDFLAGS=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
    SHARED_LDFLAGS=(-undefined dynamic_lookup)
    DEFAULT_CC=${CC:-clang}
    DEFAULT_CXX=${CXX:-clang++}
fi

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

if [ "$ENV" = "constellation" ]; then
    SRC_DIR="constellation"
    EXTRA_SRC="vendor/cJSON.c"
    OUTPUT_NAME="seethestars"
elif [ "$ENV" = "trailer" ]; then
    SRC_DIR="trailer"
    OUTPUT_NAME="trailer/trailer"
elif [ "$ENV" = "impulse_wars" ]; then
    SRC_DIR="ocean/$ENV"
    if [ "$MODE" = "web" ]; then
        BOX2D_NAME='box2d-web'
    elif [ "$PLATFORM" = "Linux" ]; then
        BOX2D_NAME='box2d-linux-amd64'
    else
        BOX2D_NAME='box2d-macos-arm64'
    fi
    BOX2D_URL="https://github.com/capnspacehook/box2d/releases/latest/download"
    download "$BOX2D_NAME" "$BOX2D_URL/$BOX2D_NAME.tar.gz"
    INCLUDES+=(-I./$BOX2D_NAME/include -I./$BOX2D_NAME/src)
    LINK_ARCHIVES+=("./$BOX2D_NAME/libbox2d.a")
elif [ -d "ocean/$ENV" ]; then
    SRC_DIR="ocean/$ENV"
else
    echo "Error: environment '$ENV' not found" && exit 1
fi

OUTPUT_NAME=${OUTPUT_NAME:-$ENV}

if [ "$MODE" = "local" ] || [ "$MODE" = "fast" ]; then
    FLAGS=(
        "${INCLUDES[@]}"
        "$SRC_DIR/$ENV.c" $EXTRA_SRC
        -o "$OUTPUT_NAME"
        "${LINK_ARCHIVES[@]}"
        "${STANDALONE_LDFLAGS[@]}"
        -DPLATFORM_DESKTOP
        -lm
    )
    if [ "$PLATFORM" = "Darwin" ]; then
        OMP_INC=$(find_omp_include || true)
        [ -n "${OMP_INC:-}" ] && FLAGS+=(-I"$OMP_INC")
        FLAGS+=(-Xclang -fopenmp)
        resolve_darwin_omp_link
        FLAGS+=("${DARWIN_OMP_LINK[@]}")
    else
        FLAGS+=(-fopenmp -lpthread)
    fi
    echo "Compiling $ENV..."
    "$DEFAULT_CC" "${CLANG_OPT[@]}" "${FLAGS[@]}"
    echo "Built: ./$OUTPUT_NAME"
    exit 0
fi

if [ "$MODE" = "web" ]; then
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

STATIC_CFLAGS=(
    "${CLANG_OPT[@]}"
    -I. -Isrc -I"$SRC_DIR" -Ivendor
    -I./"$RAYLIB_NAME"/include
    -DPLATFORM_DESKTOP
    -fvisibility=hidden
    -fPIC
)

if [ "$PLATFORM" = "Darwin" ]; then
    OMP_INC=$(find_omp_include || true)
    [ -n "${OMP_INC:-}" ] && STATIC_CFLAGS+=(-I"$OMP_INC")
    STATIC_CFLAGS+=(-Xclang -fopenmp)
else
    STATIC_CFLAGS+=(-fno-semantic-interposition)
    STATIC_CFLAGS+=(-fopenmp)
fi

echo "Compiling static library for $ENV..."
"$DEFAULT_CC" -c "${STATIC_CFLAGS[@]}" "$BINDING_SRC" -o "$STATIC_OBJ"
ar rcs "$STATIC_LIB" "$STATIC_OBJ"

if [ "$MODE" = "cpu" ]; then
    CPU_CFLAGS=(
        -c -fPIC
        -D_GLIBCXX_USE_CXX11_ABI=1
        -DPLATFORM_DESKTOP
        -DENV_NAME="$ENV"
        -std=c++17
        -I. -Isrc
        -I"$PYTHON_INCLUDE" -I"$PYBIND_INCLUDE"
        ${PRECISION:+$PRECISION}
        "${LINK_OPT[@]}"
    )
    if [ "$PLATFORM" = "Darwin" ]; then
        [ -n "${OMP_INC:-}" ] && CPU_CFLAGS+=(-I"$OMP_INC")
        CPU_CFLAGS+=(-Xclang -fopenmp)
        resolve_darwin_omp_link
    else
        CPU_CFLAGS+=(-fopenmp)
    fi

    echo "Compiling CPU training backend..."
    "$DEFAULT_CXX" "${CPU_CFLAGS[@]}" src/bindings_cpu.cpp -o build/bindings_cpu.o

    if [ "$PLATFORM" = "Darwin" ]; then
        LINK_CMD=(
            "$DEFAULT_CXX" -shared -fPIC
            build/bindings_cpu.o "$STATIC_LIB" "$RAYLIB_A"
            -lm -lpthread
            "${DARWIN_OMP_LINK[@]}"
            -framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo
            "${LINK_OPT[@]}"
            "${SHARED_LDFLAGS[@]}"
            -o "$OUTPUT"
        )
    else
        LINK_CMD=(
            "$DEFAULT_CXX" -shared -fPIC -fopenmp
            build/bindings_cpu.o "$STATIC_LIB" "$RAYLIB_A"
            -lm -lpthread "$OMP_LIB"
            "${LINK_OPT[@]}"
            "${SHARED_LDFLAGS[@]}"
            -o "$OUTPUT"
        )
    fi
    "${LINK_CMD[@]}"
    echo "Built: $OUTPUT"
    exit 0
fi

if [ "$PLATFORM" = "Darwin" ]; then
    if [ "$MODE" = "profile" ]; then
        echo "Error: --profile is only supported on the CUDA path"
        exit 1
    fi

    resolve_darwin_omp_link
    METAL_CFLAGS=(
        -c -fPIC -std=c++17 -ObjC++ -fobjc-arc
        -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
        -DPLATFORM_DESKTOP
        -DENV_NAME="$ENV"
        -DWITH_METAL
        -I. -Isrc
        -I"$PYTHON_INCLUDE" -I"$PYBIND_INCLUDE" -I"$NUMPY_INCLUDE"
        -I./"$RAYLIB_NAME"/include
        "${CLANG_OPT[@]}"
    )
    [ -n "${PRECISION:-}" ] && METAL_CFLAGS+=("$PRECISION")
    [ -n "${OMP_INC:-}" ] && METAL_CFLAGS+=(-I"$OMP_INC")
    METAL_CFLAGS+=(-Xclang -fopenmp)

    echo "Compiling Metal training backend..."
    "$DEFAULT_CXX" "${METAL_CFLAGS[@]}" src/metal_bindings.mm -o build/metal_bindings.o
    "$DEFAULT_CXX" "${METAL_CFLAGS[@]}" src/metal_platform.mm -o build/metal_platform.o

    LINK_CMD=(
        "$DEFAULT_CXX" -shared -fPIC
        build/metal_bindings.o build/metal_platform.o
        "$STATIC_LIB" "$RAYLIB_A"
        -framework Metal -framework Accelerate -framework Foundation
        -framework Cocoa -framework OpenGL -framework IOKit
        -framework CoreGraphics -framework CoreFoundation
        -framework CoreVideo -framework CoreAudio
        -framework AudioToolbox -framework UniformTypeIdentifiers
        "${DARWIN_OMP_LINK[@]}"
        "${LINK_OPT[@]}"
        "${SHARED_LDFLAGS[@]}"
        -o "$OUTPUT"
    )
    "${LINK_CMD[@]}"

    for install_name in \
        /opt/llvm-openmp/lib/libomp.dylib \
        /opt/homebrew/opt/libomp/lib/libomp.dylib \
        /usr/local/opt/libomp/lib/libomp.dylib; do
        install_name_tool -change "$install_name" "@rpath/libomp.dylib" "$OUTPUT" 2>/dev/null || true
    done

    echo "Built: $OUTPUT"
    exit 0
fi

CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-$(dirname "$(dirname "$(command -v nvcc)")")}}
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

NCCL_IFLAG=""
NCCL_LFLAG=""
for dir in /usr/include /usr/local/cuda/include; do
    if [ -f "$dir/nccl.h" ]; then
        NCCL_IFLAG="-I$dir"
        break
    fi
done
for dir in /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64; do
    if [ -f "$dir/libnccl.so" ] || [ -f "$dir/libnccl.so.2" ]; then
        NCCL_LFLAG="-L$dir"
        break
    fi
done
if [ -z "$NCCL_IFLAG" ]; then
    NCCL_IFLAG=$(python -c "import nvidia.nccl, os; print('-I' + os.path.join(nvidia.nccl.__path__[0], 'include'))" 2>/dev/null || echo "")
fi
if [ -z "$NCCL_LFLAG" ]; then
    NCCL_LFLAG=$(python -c "import nvidia.nccl, os; print('-L' + os.path.join(nvidia.nccl.__path__[0], 'lib'))" 2>/dev/null || echo "")
fi

OBS_TENSOR_T=$(awk '
    /^#define OBS_TENSOR_T/ { print $3; exit }
    /^#define OBS_TYPE/ {
        if ($3 == "FLOAT") print "FloatTensor";
        else if ($3 == "UNSIGNED_CHAR") print "ByteTensor";
        else if ($3 == "INT") print "IntTensor";
        else if ($3 == "CHAR") print "ByteTensor";
        exit
    }
' "$BINDING_SRC")
if [ -z "$OBS_TENSOR_T" ]; then
    echo "Error: Could not find OBS_TENSOR_T in $BINDING_SRC"
    exit 1
fi

export CCACHE_DIR="${CCACHE_DIR:-$HOME/.ccache}"
export CCACHE_BASEDIR="$(pwd)"
export CCACHE_COMPILERCHECK=content
NVCC="ccache $CUDA_HOME/bin/nvcc"
ARCH=${NVCC_ARCH:-native}

if [ "$MODE" = "profile" ]; then
    echo "Compiling profile binary ($ARCH)..."
    $NVCC "${NVCC_OPT[@]}" -arch=$ARCH -std=c++17 \
        -I. -Isrc -I"$SRC_DIR" -Ivendor \
        -I"$CUDA_HOME/include" $CUDNN_IFLAG $NCCL_IFLAG -I"$RAYLIB_NAME/include" \
        -DOBS_TENSOR_T="$OBS_TENSOR_T" \
        -DENV_NAME="$ENV" \
        -Xcompiler=-DPLATFORM_DESKTOP \
        ${PRECISION:+$PRECISION} \
        -Xcompiler=-fopenmp \
        tests/profile_kernels.cu vendor/ini.c \
        "$STATIC_LIB" "$RAYLIB_A" \
        -lnccl -lnvidia-ml -lcublas -lcurand -lcudnn \
        -lGL -lm -lpthread "$OMP_LIB" \
        -o profile
    echo "Built: ./profile"
    exit 0
fi

echo "Compiling CUDA ($ARCH) training backend..."
$NVCC -c -arch=$ARCH -Xcompiler -fPIC \
    -Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=1 \
    -Xcompiler=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
    -Xcompiler=-DPLATFORM_DESKTOP \
    -std=c++17 \
    -I. -Isrc \
    -I"$PYTHON_INCLUDE" -I"$PYBIND_INCLUDE" -I"$NUMPY_INCLUDE" \
    -I"$CUDA_HOME/include" $CUDNN_IFLAG $NCCL_IFLAG -I"$RAYLIB_NAME/include" \
    -Xcompiler=-fopenmp \
    -DOBS_TENSOR_T="$OBS_TENSOR_T" \
    -DENV_NAME="$ENV" \
    ${PRECISION:+$PRECISION} "${NVCC_OPT[@]}" \
    src/bindings.cu -o build/bindings.o

LINK_CMD=(
    "$DEFAULT_CXX" -shared -fPIC -fopenmp
    build/bindings.o "$STATIC_LIB" "$RAYLIB_A"
    -L"$CUDA_HOME/lib64" $CUDNN_LFLAG $NCCL_LFLAG
    -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand -lcudnn
    "$OMP_LIB"
    "${LINK_OPT[@]}"
    "${SHARED_LDFLAGS[@]}"
    -o "$OUTPUT"
)
"${LINK_CMD[@]}"
echo "Built: $OUTPUT"
