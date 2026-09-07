#!/bin/bash
# build-hip.sh - AMD ROCm fused trainer (_C.so) via HIPIFY-translated sources
set -e
ENV=${1:?usage: build-hip.sh ENV [cartpole|breakout|...]}
cd "$(dirname "$0")"

SRC_DIR="ocean/$ENV"
RAYLIB_NAME='raylib-5.5_linux_amd64'
if [ ! -d "$RAYLIB_NAME" ]; then
    curl -sL "https://github.com/raysan5/raylib/releases/download/5.5/$RAYLIB_NAME.tar.gz" -o r.tgz
    tar xf r.tgz && rm r.tgz
fi

HIPCC=/opt/rocm/bin/hipcc
ARCH=${HIP_ARCH:-gfx1101}
ROCM=/opt/rocm

PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INCLUDE=$(python -c "import pybind11; print(pybind11.get_include())")
NUMPY_INCLUDE=$(python -c "import numpy; print(numpy.get_include())")
EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTPUT="pufferlib/_C${EXT_SUFFIX}"

BINDING_SRC="$SRC_DIR/binding.c"
mkdir -p build
STATIC_LIB="build/libstatic_${ENV}.a"
clang -c -O2 -std=gnu11 -D_GNU_SOURCE -I. -Isrc -I"$SRC_DIR" -Ivendor \
    -I./$RAYLIB_NAME/include -DPLATFORM_DESKTOP \
    -fno-semantic-interposition -fvisibility=hidden -fPIC -fopenmp \
    "$BINDING_SRC" -o build/libstatic_${ENV}.o
ar rcs "$STATIC_LIB" build/libstatic_${ENV}.o
OBS_TENSOR_T=$(awk '/^#define OBS_TENSOR_T/{print $3}' "$BINDING_SRC")

echo "Compiling HIP training backend ($ARCH)..."
"$HIPCC" -c --offload-arch=$ARCH -fPIC -std=c++17 -O2 \
    -I. -Isrc -Isrc-hip -Isrc-hip/stub \
    -I"$PYTHON_INCLUDE" -I"$PYBIND_INCLUDE" -I"$NUMPY_INCLUDE" \
    -I$ROCM/include -I$ROCM/include/hipblas -I$ROCM/include/hiprand -I./$RAYLIB_NAME/include \
    -D_GLIBCXX_USE_CXX11_ABI=1 -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
    -DPLATFORM_DESKTOP -fopenmp \
    -DOBS_TENSOR_T=$OBS_TENSOR_T -DENV_NAME=$ENV -DPRECISION_FLOAT \
    src-hip/bindings.hip.cpp -o build/bindings_hip.o

"$HIPCC" -c -fPIC -std=c++17 -O2 -x hip --offload-arch=$ARCH \
    src-hip/cuda_shim.cpp -o build/cuda_shim.o
g++ -shared -fPIC -fopenmp \
    build/bindings_hip.o build/cuda_shim.o "$STATIC_LIB" "$RAYLIB_NAME/lib/libraylib.a" \
    -L$ROCM/lib -lamdhip64 -lhipblas -lhiprand -lrocrand -lrccl -lMIOpen \
    -Wl,-rpath,$ROCM/lib -Bsymbolic-functions \
    -lm -lpthread -lomp5 \
    -o "$OUTPUT"
echo "Built: $OUTPUT"
