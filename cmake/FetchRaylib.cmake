# Acquire prebuilt raylib 5.5 release binaries (same pinned archives build.sh used)
# and expose them as the imported target raylib::raylib.
#
# Override with -DPUFFER_RAYLIB_ROOT=<dir> to use a local copy (offline builds).
# <dir> must contain include/raylib.h and lib/libraylib.a (or raylib.lib on Windows).

include(FetchContent)

set(PUFFER_RAYLIB_ROOT "" CACHE PATH "Local raylib root (include/ + lib/), skips download")

set(_RAYLIB_VERSION "5.5")
set(_RAYLIB_URL_BASE "https://github.com/raysan5/raylib/releases/download/${_RAYLIB_VERSION}")

if(PUFFER_RAYLIB_ROOT)
    set(_raylib_dir "${PUFFER_RAYLIB_ROOT}")
else()
    if(EMSCRIPTEN)
        set(_raylib_name "raylib-${_RAYLIB_VERSION}_webassembly")
        set(_raylib_ext "zip")
    elseif(WIN32)
        set(_raylib_name "raylib-${_RAYLIB_VERSION}_win64_msvc16")
        set(_raylib_ext "zip")
    elseif(APPLE)
        set(_raylib_name "raylib-${_RAYLIB_VERSION}_macos")
        set(_raylib_ext "tar.gz")
    else()
        set(_raylib_name "raylib-${_RAYLIB_VERSION}_linux_amd64")
        set(_raylib_ext "tar.gz")
    endif()

    FetchContent_Declare(raylib_bin
        URL "${_RAYLIB_URL_BASE}/${_raylib_name}.${_raylib_ext}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(raylib_bin)
    set(_raylib_dir "${raylib_bin_SOURCE_DIR}")
endif()

find_library(PUFFER_RAYLIB_LIBRARY
    NAMES raylib libraylib.a
    PATHS "${_raylib_dir}/lib"
    NO_DEFAULT_PATH
)
if(NOT PUFFER_RAYLIB_LIBRARY OR NOT EXISTS "${_raylib_dir}/include/raylib.h")
    message(FATAL_ERROR "raylib not found under ${_raylib_dir} (expected include/raylib.h and lib/)")
endif()

add_library(raylib::raylib STATIC IMPORTED)
set_target_properties(raylib::raylib PROPERTIES
    IMPORTED_LOCATION "${PUFFER_RAYLIB_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${_raylib_dir}/include"
)

# Platform link requirements of the static raylib
if(WIN32)
    set_property(TARGET raylib::raylib APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES winmm gdi32 opengl32 user32 shell32)
elseif(APPLE)
    set_property(TARGET raylib::raylib APPEND PROPERTY
        INTERFACE_LINK_OPTIONS "SHELL:-framework Cocoa" "SHELL:-framework IOKit" "SHELL:-framework CoreVideo")
elseif(NOT EMSCRIPTEN)
    set_property(TARGET raylib::raylib APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES m pthread)
endif()
