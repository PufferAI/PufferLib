# FindNCCL
# --------
# Locates NCCL (multi-GPU collectives). NCCL has no Windows build, so this
# module reports NOT FOUND on Windows unconditionally; the build then defines
# no PUFFER_HAS_NCCL and multi-GPU support is compiled out.
#
# Search order on Linux: NCCL_ROOT env/cache, system dirs, CUDA toolkit dirs,
# then the nvidia-nccl pip wheel in the active Python environment.
#
# Result variables / targets:
#   NCCL_FOUND, NCCL_INCLUDE_DIR, NCCL_LIBRARY, target NCCL::nccl

if(WIN32 OR EMSCRIPTEN)
    set(NCCL_FOUND FALSE)
    if(NCCL_FIND_REQUIRED)
        message(FATAL_ERROR "NCCL is not available on this platform (Linux only)")
    endif()
    return()
endif()

set(_nccl_hints "")
foreach(_v NCCL_ROOT NCCL_PATH)
    if(DEFINED ${_v})
        list(APPEND _nccl_hints "${${_v}}")
    endif()
    if(DEFINED ENV{${_v}})
        list(APPEND _nccl_hints "$ENV{${_v}}")
    endif()
endforeach()
if(CUDAToolkit_FOUND)
    list(APPEND _nccl_hints "${CUDAToolkit_LIBRARY_ROOT}")
endif()

find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${_nccl_hints} PATH_SUFFIXES include)
find_library(NCCL_LIBRARY NAMES nccl HINTS ${_nccl_hints} PATH_SUFFIXES lib lib64)

# Pip wheel fallback (nvidia-nccl-cu12)
if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__))"
        OUTPUT_VARIABLE _nccl_wheel_dir OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET RESULT_VARIABLE _nccl_wheel_rc
    )
    if(_nccl_wheel_rc EQUAL 0 AND _nccl_wheel_dir)
        find_path(NCCL_INCLUDE_DIR nccl.h HINTS "${_nccl_wheel_dir}" PATH_SUFFIXES include)
        find_library(NCCL_LIBRARY NAMES nccl HINTS "${_nccl_wheel_dir}" PATH_SUFFIXES lib)
        if(NCCL_LIBRARY)
            get_filename_component(NCCL_WHEEL_LIB_DIR "${NCCL_LIBRARY}" DIRECTORY)
            set(NCCL_FROM_WHEEL TRUE)
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
    REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
)

if(NCCL_FOUND AND NOT TARGET NCCL::nccl)
    add_library(NCCL::nccl UNKNOWN IMPORTED)
    set_target_properties(NCCL::nccl PROPERTIES
        IMPORTED_LOCATION "${NCCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
