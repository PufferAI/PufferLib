# FindCUDNN
# ---------
# Locates cuDNN. Search order:
#   1. CUDNN_ROOT / CUDNN_PATH (cache or environment; covers the NVIDIA Windows
#      installer default C:/Program Files/NVIDIA/CUDNN/v9.x and manual unpacks)
#   2. The CUDA toolkit directories (typical Linux system installs)
#   3. The nvidia-cudnn pip wheel in the active Python environment.
#      NOTE: only useful for LINKING on Linux (the win_amd64 wheel ships DLLs
#      and headers but no .lib import libraries).
#   4. Windows only, opt-in: -DPUFFER_FETCH_CUDNN=ON downloads the NVIDIA
#      redistributable archive (~1.7 GB) into the build tree.
#
# Result variables / targets:
#   CUDNN_FOUND, CUDNN_INCLUDE_DIR, CUDNN_LIBRARY, target CUDNN::cudnn

option(PUFFER_FETCH_CUDNN "Download the NVIDIA cuDNN redist archive if not found locally (Windows, large download)" OFF)

set(_cudnn_hints "")
foreach(_v CUDNN_ROOT CUDNN_PATH)
    if(DEFINED ${_v})
        list(APPEND _cudnn_hints "${${_v}}")
    endif()
    if(DEFINED ENV{${_v}})
        list(APPEND _cudnn_hints "$ENV{${_v}}")
    endif()
endforeach()

# Windows installer layout: C:/Program Files/NVIDIA/CUDNN/v9.x/{include,lib}/<cuda-ver>/
if(WIN32)
    file(GLOB _cudnn_prog_dirs "$ENV{ProgramFiles}/NVIDIA/CUDNN/v9*")
    list(APPEND _cudnn_hints ${_cudnn_prog_dirs})
endif()

if(CUDAToolkit_FOUND)
    list(APPEND _cudnn_hints "${CUDAToolkit_LIBRARY_ROOT}" "${CUDAToolkit_LIBRARY_DIR}/..")
endif()

macro(_puffer_cudnn_search)
    find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS ${_cudnn_hints}
        PATH_SUFFIXES include include/12.9 include/12.8 include/13.0
    )
    find_library(CUDNN_LIBRARY
        NAMES cudnn cudnn9
        HINTS ${_cudnn_hints}
        PATH_SUFFIXES lib lib64 lib/x64 lib/12.9/x64 lib/12.8/x64 lib/13.0/x64
    )
endmacro()

_puffer_cudnn_search()

# Pip wheel fallback (Linux linking; on Windows headers only, so skip for linking)
if((NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY) AND NOT WIN32)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))"
        OUTPUT_VARIABLE _cudnn_wheel_dir OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET RESULT_VARIABLE _cudnn_wheel_rc
    )
    if(_cudnn_wheel_rc EQUAL 0 AND _cudnn_wheel_dir)
        list(APPEND _cudnn_hints "${_cudnn_wheel_dir}")
        _puffer_cudnn_search()
        if(CUDNN_LIBRARY)
            # Consumers add an rpath so the wheel .so resolves at runtime
            get_filename_component(CUDNN_WHEEL_LIB_DIR "${CUDNN_LIBRARY}" DIRECTORY)
            set(CUDNN_FROM_WHEEL TRUE)
        endif()
    endif()
endif()

# Windows opt-in redist download (includes the .lib import libraries the wheel lacks)
if((NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY) AND WIN32 AND PUFFER_FETCH_CUDNN)
    include(FetchContent)
    set(_cudnn_redist_ver "9.23.2.1")  # verified present in the redist index, cuda12 variant
    FetchContent_Declare(cudnn_redist
        URL "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-${_cudnn_redist_ver}_cuda12-archive.zip"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    message(STATUS "Downloading cuDNN ${_cudnn_redist_ver} redist archive (large download, one-time)...")
    FetchContent_MakeAvailable(cudnn_redist)
    list(APPEND _cudnn_hints "${cudnn_redist_SOURCE_DIR}")
    _puffer_cudnn_search()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN
    REQUIRED_VARS CUDNN_LIBRARY CUDNN_INCLUDE_DIR
    REASON_FAILURE_MESSAGE
    "cuDNN not found. Options: set CUDNN_ROOT to an SDK install. On Linux 'pip install nvidia-cudnn-cu12'. On Windows install cuDNN from https://developer.nvidia.com/cudnn or configure with -DPUFFER_FETCH_CUDNN=ON (large download). Note: the Windows pip wheel has no import libraries and cannot be used for building."
)

if(CUDNN_FOUND AND NOT TARGET CUDNN::cudnn)
    add_library(CUDNN::cudnn UNKNOWN IMPORTED)
    set_target_properties(CUDNN::cudnn PROPERTIES
        IMPORTED_LOCATION "${CUDNN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)
