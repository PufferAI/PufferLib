cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(snakegame)

# Find PyTorch package
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "./libtorch")
find_package(Torch REQUIRED)

# Set Raylib path
set(RAYLIB_PATH "./raylib-5.0_linux_amd64")

# Include directories
include_directories(${RAYLIB_PATH}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# Find Raylib library
find_library(RAYLIB_LIB NAMES raylib HINTS ${RAYLIB_PATH}/lib NO_CMAKE_FIND_ROOT_PATH)

if(NOT RAYLIB_LIB)
    message(FATAL_ERROR "Raylib library not found")
endif()

# Add executable
add_executable(snakegame claude_snake_torch.cpp)

# Set C++ standard
set_property(TARGET snakegame PROPERTY CXX_STANDARD 17)

# Add compiler flags
target_compile_options(snakegame PRIVATE -g)

# Link libraries
target_link_libraries(snakegame
    ${TORCH_LIBRARIES}
    ${RAYLIB_LIB}
    GL
    m
    pthread
    dl
    rt
    X11
)

# Set runtime path for shared libraries
set_target_properties(snakegame PROPERTIES
    INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/raylib-5.0_linux_amd64/lib;${CMAKE_PREFIX_PATH}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Print some debug information
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "RAYLIB_LIB: ${RAYLIB_LIB}")
