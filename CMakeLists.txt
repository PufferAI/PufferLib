cmake_minimum_required(VERSION 3.24)

# PufferLib native build.
#
# One environment per configure:
#   cmake --preset cuda -DENV=breakout && cmake --build --preset cuda
# or use ./build.sh (Linux/macOS) / ./build.ps1 (Windows) which map the
# historical flags onto presets. See CMakePresets.json for the full list.
#
# Modes (PUFFER_MODE):
#   cuda        pufferlib/_C extension with the CUDA training backend (default)
#   cpu         pufferlib/_C extension, CPU-only training backend
#   standalone  self-contained environment executable (local/fast presets)
#   web         emscripten build (requires EMSDK toolchain, web preset)
#   profile     kernel profiling binary (Linux)

project(pufferlib LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(ENV "" CACHE STRING "Environment to build (e.g. breakout); see ocean/")
set(PUFFER_MODE "cuda" CACHE STRING "Build mode: cuda | cpu | standalone | web | profile")
option(PRECISION_FLOAT "Use float32 precision instead of bf16 (forced ON for cpu mode)" OFF)
option(PUFFER_SANITIZE "Enable ASan/UBSan for standalone builds (non-Windows)" OFF)
option(PUFFER_NO_OPENMP "Disable OpenMP in the environment vectorizer" OFF)

if(NOT ENV)
    message(FATAL_ERROR "Set -DENV=<name> (e.g. -DENV=breakout). Available: ocean/* directories, constellation, trailer")
endif()
if(PUFFER_MODE STREQUAL "cpu")
    set(PRECISION_FLOAT ON)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Parity with build.sh optimization levels (-O2, not CMake's -O3 default)
if(NOT MSVC)
    set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
endif()

if(WIN32)
    # UCRT: expose M_PI etc., silence fopen/strcpy deprecation spam
    add_compile_definitions(_USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)

    # Help FindOpenMP locate LLVM's libomp next to clang (the GNU-driver clang
    # does not add its own lib dir to the linker search path).
    if(CMAKE_C_COMPILER_ID STREQUAL "Clang" OR NOT CMAKE_C_COMPILER_ID)
        find_program(_puffer_clang clang)
        if(_puffer_clang)
            get_filename_component(_llvm_bin "${_puffer_clang}" DIRECTORY)
            find_library(OpenMP_libomp_LIBRARY NAMES libomp HINTS "${_llvm_bin}/../lib")
            if(OpenMP_libomp_LIBRARY)
                set(OpenMP_C_FLAGS "-fopenmp=libomp" CACHE STRING "")
                set(OpenMP_C_LIB_NAMES "libomp" CACHE STRING "")
                set(PUFFER_LIBOMP_DLL "${_llvm_bin}/libomp.dll")
            endif()
        endif()
    endif()
endif()

# ccache when available (content-based compiler check like build.sh did)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    if(MSVC)
        set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug>:Embedded>")
    endif()
endif()

# ----------------------------------------------------------------------------
# Environment source resolution (mirrors the old build.sh special cases)
# ----------------------------------------------------------------------------
set(EXTRA_SRC "")
set(OUTPUT_NAME "${ENV}")
set(EXE_OUTPUT_DIR "${CMAKE_SOURCE_DIR}")
set(ENV_EXTRA_LIBS "")
set(ENV_EXTRA_INCLUDES "")

set(PUFFER_WIN_UNSUPPORTED nethack impulse_wars boxoban chess trailer)
if(WIN32 AND ENV IN_LIST PUFFER_WIN_UNSUPPORTED)
    message(FATAL_ERROR
        "Environment '${ENV}' is not yet supported on Windows "
        "(POSIX-only dependencies: nethack=NLE/memfd, impulse_wars=box2d prebuilt, "
        "boxoban=mmap, chess=fork/exec engine, trailer=fork/wait). "
        "Build it on Linux/macOS/WSL.")
endif()

if(ENV STREQUAL "constellation")
    set(SRC_DIR "${CMAKE_SOURCE_DIR}/constellation")
    set(EXTRA_SRC "${CMAKE_SOURCE_DIR}/vendor/cJSON.c")
    set(OUTPUT_NAME "seethestars")
elseif(ENV STREQUAL "trailer")
    set(SRC_DIR "${CMAKE_SOURCE_DIR}/trailer")
    set(EXE_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/trailer")
elseif(ENV STREQUAL "impulse_wars")
    set(SRC_DIR "${CMAKE_SOURCE_DIR}/ocean/${ENV}")
    include(FetchContent)
    if(EMSCRIPTEN)
        set(_box2d_name "box2d-web")
    elseif(APPLE)
        set(_box2d_name "box2d-macos-arm64")
    else()
        set(_box2d_name "box2d-linux-amd64")
    endif()
    FetchContent_Declare(box2d_bin
        URL "https://github.com/capnspacehook/box2d/releases/latest/download/${_box2d_name}.tar.gz"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(box2d_bin)
    list(APPEND ENV_EXTRA_INCLUDES "${box2d_bin_SOURCE_DIR}/include" "${box2d_bin_SOURCE_DIR}/src")
    list(APPEND ENV_EXTRA_LIBS "${box2d_bin_SOURCE_DIR}/libbox2d.a")
elseif(ENV STREQUAL "nethack")
    set(SRC_DIR "${CMAKE_SOURCE_DIR}/ocean/${ENV}")
    set(NLE_DIR "${CMAKE_SOURCE_DIR}/vendor/nle")
    if(NOT EXISTS "${NLE_DIR}/src")
        message(STATUS "Cloning modified NLE...")
        execute_process(
            COMMAND git clone --depth 1 https://github.com/liujonathan24/NetHack.git "${NLE_DIR}"
            RESULT_VARIABLE _nle_rc)
        if(NOT _nle_rc EQUAL 0)
            message(FATAL_ERROR "Failed to clone NLE for nethack")
        endif()
    endif()
    set(NETHACK_LIB_DIR "${NLE_DIR}/src/build")
    if(NOT EXISTS "${NETHACK_LIB_DIR}/libnethack.so")
        message(STATUS "Building libnethack.so ...")
        include(ProcessorCount)
        ProcessorCount(_nproc)
        execute_process(
            COMMAND make -C "${NETHACK_LIB_DIR}" nethack -j${_nproc}
            RESULT_VARIABLE _nh_rc)
        if(NOT _nh_rc EQUAL 0)
            message(FATAL_ERROR "Failed to build libnethack.so")
        endif()
    endif()
    list(APPEND ENV_EXTRA_INCLUDES "${NLE_DIR}/include")
    list(APPEND ENV_EXTRA_LIBS "-L${NETHACK_LIB_DIR}" nethack "-Wl,-rpath,${NETHACK_LIB_DIR}" dl)
elseif(EXISTS "${CMAKE_SOURCE_DIR}/ocean/${ENV}")
    set(SRC_DIR "${CMAKE_SOURCE_DIR}/ocean/${ENV}")
else()
    message(FATAL_ERROR "Environment '${ENV}' not found (no ocean/${ENV} directory)")
endif()

include(FetchRaylib)

set(PUFFER_INCLUDES
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/src"
    "${SRC_DIR}"
    "${CMAKE_SOURCE_DIR}/vendor"
    ${ENV_EXTRA_INCLUDES}
)

# Warning set carried over from build.sh (clang/gcc only)
set(PUFFER_C_WARNINGS
    -Wall
    -ferror-limit=3
    -Werror=incompatible-pointer-types
    -Werror=return-type
    -Wno-error=incompatible-pointer-types-discards-qualifiers
    -Wno-incompatible-pointer-types-discards-qualifiers
    -Wno-error=array-parameter
)
# x86_64 SIMD used directly by drive.h and src/bf16.h
set(PUFFER_SIMD_FLAGS -mavx2 -mfma)

set(PUFFER_SANITIZE_FLAGS "")
if(PUFFER_SANITIZE AND NOT WIN32 AND NOT APPLE AND NOT EMSCRIPTEN)
    set(PUFFER_SANITIZE_FLAGS -fsanitize=address,undefined,bounds,pointer-overflow,leak -fno-omit-frame-pointer)
endif()

# ============================================================================
# Web build (emscripten toolchain; separate preset)
# ============================================================================
if(EMSCRIPTEN)
    if(NOT PUFFER_MODE STREQUAL "web")
        message(FATAL_ERROR "Emscripten toolchain requires -DPUFFER_MODE=web")
    endif()
    add_executable(game "${SRC_DIR}/${ENV}.c" ${EXTRA_SRC})
    target_include_directories(game PRIVATE ${PUFFER_INCLUDES})
    target_compile_options(game PRIVATE -O3 -Wall)
    target_compile_definitions(game PRIVATE NDEBUG PLATFORM_WEB GRAPHICS_API_OPENGL_ES3)
    target_link_libraries(game PRIVATE raylib::raylib ${ENV_EXTRA_LIBS})
    target_link_options(game PRIVATE
        "-sASSERTIONS=2" "-gsource-map"
        "-sUSE_GLFW=3" "-sUSE_WEBGL2=1" "-sASYNCIFY" "-sFILESYSTEM" "-sFORCE_FILESYSTEM=1"
        "--shell-file=${CMAKE_SOURCE_DIR}/vendor/minshell.html"
        "-sINITIAL_MEMORY=512MB" "-sALLOW_MEMORY_GROWTH" "-sSTACK_SIZE=512KB"
        "--preload-file=${CMAKE_SOURCE_DIR}/resources/${ENV}@resources/${ENV}"
        "--preload-file=${CMAKE_SOURCE_DIR}/resources/shared@resources/shared"
    )
    set_target_properties(game PROPERTIES
        SUFFIX ".html"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/web/${ENV}"
    )
    set_property(TARGET game PROPERTY LINK_DEPENDS "${CMAKE_SOURCE_DIR}/vendor/minshell.html")
    return()
endif()

# ============================================================================
# Standalone environment executable (local/fast presets)
# ============================================================================
if(PUFFER_MODE STREQUAL "standalone")
    add_executable(env_exe "${SRC_DIR}/${ENV}.c" ${EXTRA_SRC})
    target_include_directories(env_exe PRIVATE ${PUFFER_INCLUDES})
    target_compile_definitions(env_exe PRIVATE PLATFORM_DESKTOP)
    if(WIN32)
        # Env code assumes glibc niceties (rand_r, M_PI, usleep, ...); the shim
        # provides them without per-file edits.
        target_compile_options(env_exe PRIVATE "SHELL:-include ${CMAKE_SOURCE_DIR}/src/puffer_os.h")
    endif()
    target_compile_options(env_exe PRIVATE
        ${PUFFER_C_WARNINGS} ${PUFFER_SIMD_FLAGS} ${PUFFER_SANITIZE_FLAGS})
    target_link_options(env_exe PRIVATE ${PUFFER_SANITIZE_FLAGS})
    target_link_libraries(env_exe PRIVATE raylib::raylib ${ENV_EXTRA_LIBS})
    if(NOT WIN32)
        target_link_libraries(env_exe PRIVATE m pthread)
    endif()
    if(UNIX AND NOT APPLE)
        target_link_libraries(env_exe PRIVATE GL)
    endif()
    if(NOT PUFFER_NO_OPENMP)
        find_package(OpenMP COMPONENTS C)
        if(OpenMP_C_FOUND)
            target_link_libraries(env_exe PRIVATE OpenMP::OpenMP_C)
        endif()
    endif()
    set_target_properties(env_exe PROPERTIES
        OUTPUT_NAME "${OUTPUT_NAME}"
        RUNTIME_OUTPUT_DIRECTORY "${EXE_OUTPUT_DIR}"
    )
    return()
endif()

# ============================================================================
# Python extension modes (cuda / cpu) and profile
# ============================================================================
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PY_EXT_SUFFIX OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(pybind11 CONFIG REQUIRED)

if(PUFFER_MODE STREQUAL "cuda" OR PUFFER_MODE STREQUAL "profile")
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O2 --threads 0 -DNDEBUG")
    find_package(CUDAToolkit REQUIRED)
    find_package(CUDNN REQUIRED)
    find_package(NCCL)  # optional; absent on Windows -> multi-GPU compiled out
endif()

# Extract OBS_TENSOR_T from binding.c (replaces the old awk hack)
set(BINDING_SRC "${SRC_DIR}/binding.c")
if(NOT EXISTS "${BINDING_SRC}")
    message(FATAL_ERROR "${BINDING_SRC} not found")
endif()
file(READ "${BINDING_SRC}" _binding_content)
string(REGEX MATCH "#define[ \t]+OBS_TENSOR_T[ \t]+([A-Za-z0-9_]+)" _obs_match "${_binding_content}")
if(NOT CMAKE_MATCH_1)
    message(FATAL_ERROR "Could not find '#define OBS_TENSOR_T <type>' in ${BINDING_SRC}")
endif()
set(OBS_TENSOR_T "${CMAKE_MATCH_1}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${BINDING_SRC}")

# ---- Static environment library (C, compiled with clang on all platforms) ----
add_library(env_static STATIC "${BINDING_SRC}")
target_include_directories(env_static PRIVATE ${PUFFER_INCLUDES})
target_compile_definitions(env_static PRIVATE PLATFORM_DESKTOP)
target_compile_options(env_static PRIVATE
    ${PUFFER_C_WARNINGS} ${PUFFER_SIMD_FLAGS}
    -fvisibility=hidden
)
if(NOT WIN32)
    target_compile_options(env_static PRIVATE -fno-semantic-interposition)
    set_target_properties(env_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
if(WIN32)
    target_compile_options(env_static PRIVATE "SHELL:-include ${CMAKE_SOURCE_DIR}/src/puffer_os.h")
endif()
target_link_libraries(env_static PRIVATE raylib::raylib)  # include dirs

if(WIN32)
    # The extension runs inside a PyTorch process, and torch's Windows wheels
    # ship Intel's OpenMP runtime (libiomp5md.dll). Loading LLVM's libomp.dll
    # alongside it aborts with OMP Error #15, so the extension's env code is
    # built without OpenMP: the parallel-for pragmas degrade to serial and
    # vecenv's own buffer threads still provide parallelism.
    set(PUFFER_NO_OPENMP ON)
    message(STATUS "OpenMP disabled for the Python extension on Windows (torch ships its own OpenMP runtime)")
endif()
if(NOT PUFFER_NO_OPENMP)
    find_package(OpenMP COMPONENTS C)
    if(OpenMP_C_FOUND)
        target_link_libraries(env_static PUBLIC OpenMP::OpenMP_C)
    else()
        message(WARNING "OpenMP for C not found; environment stepping will be serial")
        target_compile_definitions(env_static PRIVATE PUFFER_NO_OPENMP)
    endif()
else()
    target_compile_definitions(env_static PRIVATE PUFFER_NO_OPENMP)
endif()

# ---- Profile binary (Linux) ----
if(PUFFER_MODE STREQUAL "profile")
    add_executable(profile
        "${CMAKE_SOURCE_DIR}/tests/profile_kernels.cu"
        "${CMAKE_SOURCE_DIR}/vendor/ini.c")
    target_include_directories(profile PRIVATE ${PUFFER_INCLUDES})
    target_compile_definitions(profile PRIVATE
        OBS_TENSOR_T=${OBS_TENSOR_T} ENV_NAME=${ENV} PLATFORM_DESKTOP
        $<$<BOOL:${PRECISION_FLOAT}>:PRECISION_FLOAT>
        $<$<TARGET_EXISTS:NCCL::nccl>:PUFFER_HAS_NCCL>)
    target_link_libraries(profile PRIVATE
        env_static raylib::raylib ${ENV_EXTRA_LIBS}
        CUDA::cudart CUDA::cublas CUDA::curand CUDA::nvml CUDNN::cudnn
        GL m pthread)
    if(TARGET NCCL::nccl)
        target_link_libraries(profile PRIVATE NCCL::nccl)
    endif()
    set_target_properties(profile PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
    return()
endif()

# ---- _C Python extension ----
if(PUFFER_MODE STREQUAL "cuda")
    set(BINDINGS_SRC "${CMAKE_SOURCE_DIR}/src/bindings.cu")
else()
    set(BINDINGS_SRC "${CMAKE_SOURCE_DIR}/src/bindings_cpu.cpp")
endif()

add_library(_C MODULE "${BINDINGS_SRC}")
target_include_directories(_C PRIVATE "${CMAKE_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(_C PRIVATE
    pybind11::module Python3::Module Python3::NumPy
    env_static raylib::raylib ${ENV_EXTRA_LIBS})
target_compile_definitions(_C PRIVATE
    OBS_TENSOR_T=${OBS_TENSOR_T}
    ENV_NAME=${ENV}
    PLATFORM_DESKTOP
    NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
    $<$<BOOL:${PRECISION_FLOAT}>:PRECISION_FLOAT>
)
set_target_properties(_C PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_C"
    SUFFIX "${PY_EXT_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pufferlib"
    POSITION_INDEPENDENT_CODE ON
)


if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # GCC libstdc++ ABI pin + symbol binding, as in build.sh
    target_compile_definitions(_C PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    target_link_options(_C PRIVATE "LINKER:-Bsymbolic-functions")
endif()

if(PUFFER_MODE STREQUAL "cuda")
    target_link_libraries(_C PRIVATE
        CUDA::cublas CUDA::cusolver CUDA::curand CUDA::nvml CUDNN::cudnn)
    if(WIN32)
        target_link_libraries(_C PRIVATE CUDA::cudart_static)
    else()
        target_link_libraries(_C PRIVATE CUDA::cudart)
    endif()
    if(TARGET NCCL::nccl)
        target_compile_definitions(_C PRIVATE PUFFER_HAS_NCCL)
        target_link_libraries(_C PRIVATE NCCL::nccl)
    endif()
    # rpaths so pip-wheel-provided cudnn/nccl resolve at runtime (Linux)
    foreach(_wheel_dir IN LISTS CUDNN_WHEEL_LIB_DIR NCCL_WHEEL_LIB_DIR)
        if(_wheel_dir)
            target_link_options(_C PRIVATE "LINKER:-rpath,${_wheel_dir}")
        endif()
    endforeach()
    if(WIN32)
        # Host (cl.exe) side of nvcc compilation
        target_compile_options(_C PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc -Xcompiler=/Zc:__cplusplus>)
    endif()
else()
    if(NOT WIN32)
        target_link_libraries(_C PRIVATE m pthread)
    endif()
endif()
