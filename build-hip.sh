#!/usr/bin/env bash
set -euo pipefail

# Linux/ROCm: ./build-hip.sh cartpole [--cu] [extra hipcc flags...]
# Requires ROCm (hipblas, hiprand, hipsolver, RCCL) and raylib 5.5.
# Run ./build/hip/puffer train|eval|match|sweep with the usual CLI options.
cd "$(dirname "$0")"
env=${1:?usage: build-hip.sh ENV [--cu] [extra hipcc flags...]}
shift
suffix=h
if [[ ${1:-} == --cu ]]; then suffix=cu; shift; fi
rocm=${ROCM_PATH:-/opt/rocm}
raylib=${RAYLIB_PATH:-raylib-5.5_linux_amd64}
build=build/hip
[[ -f ocean/$env/$env.$suffix ]] || { echo "Missing ocean/$env/$env.$suffix" >&2; exit 1; }
[[ -f $raylib/lib/libraylib.a ]] || { echo "Set RAYLIB_PATH to raylib 5.5" >&2; exit 1; }

# Translate current sources on every build. Generated files stay in build/.
for source in src/*.cu src/*.cuh src/*.h ocean/"$env"/*.cu ocean/"$env"/*.h; do
    [[ -f $source ]] || continue
    mkdir -p "$build/$(dirname "$source")"
    "$rocm/bin/hipify-perl" --quiet-warnings "$source" > "$build/$source"
done

# HIPIFY does not cover NVML/NVTX, this PTX timer, or these API differences.
sed -i \
    -e '/nvml/d' -e '/nvtx/d' \
    -e '/\/\/ C standard/i\__device__ inline __hip_bfloat16 __ldg(const __hip_bfloat16* p) { return *p; }' \
    -e '/dict_set(out, "util\/gpu_percent"/d' \
    -e '/snprintf(gpu, /c\    snprintf(gpu, sizeof(gpu), "n/a");' \
    -e 's@<nccl.h>@<rccl/rccl.h>@' \
    -e 's/hipGraphInstantiate(/hipGraphInstantiateWithFlags(/g' \
    -e '/asm volatile.*globaltimer/c\    t = wall_clock64();' \
    -e '/unsigned long long h\[NUM_TE\];/a\        int clock_khz;\n        assert(hipDeviceGetAttribute(\&clock_khz, hipDeviceAttributeWallClockRate, hypers->gpu_id) == hipSuccess);' \
    -e 's/\* 1e-6f;/\/ clock_khz;/' \
    "$build/src/pufferl.cu"
# hipBLAS manages its workspace internally.
sed -i '/hipblasSetWorkspace/d; /hipMalloc(workspace, ws_bytes)/d' "$build/src/algo.cu"

"$rocm/bin/hipcc" --offload-arch="${HIP_ARCH:-native}" -std=c++17 -O2 -fopenmp \
    -Wno-narrowing -Wno-deprecated-declarations \
    -I"$build" -I"$build/src" -I. -Isrc -Ivendor -I"$raylib/include" \
    -I"$rocm/include/hipblas" -I"$rocm/include/hiprand" -I"$rocm/include/hipsolver" \
    -DENV_HEADER=\"ocean/$env/$env.$suffix\" -DENV_NAME="$env" \
    -DPUFFER_ENV_NAME=\"$env\" -DPUFFER_"${env^^}" -DPUFFERLIB_BUILD_MAIN \
    -DPLATFORM_DESKTOP \
    -x hip "$build/src/pufferl.cu" -x none "$raylib/lib/libraylib.a" \
    -L"$rocm/lib" -Wl,-rpath,"$rocm/lib" \
    -lhipblas -lhiprand -lrocrand -lhipsolver -lrccl -lGL -lm -lpthread \
    "$@" -o "$build/puffer"
printf 'Built %s/puffer\n' "$build"
