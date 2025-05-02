# 3.22 was released on Nov 2021, should be widely available
cmake_minimum_required(VERSION 3.22)
include(FetchContent)

project(
	impulse-wars
	DESCRIPTION "Impulse Wars"
	LANGUAGES C
)

message(INFO " C Compiler: ${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_VERSION} ${CMAKE_C_COMPILER_ID}")

# use ccache if available to speed up subsequent builds
find_program(CCACHE_FOUND "ccache")
if(CCACHE_FOUND)
	set(CMAKE_C_COMPILER_LAUNCHER "ccache")
endif()

# enable some C23 features, the c2x standard is a WIP standard supported
# by gcc since 9 (May 2019) and clang since 9 (Sep 2019)
set(CMAKE_C_FLAGS_INIT " -std=c2x")

# force position independent code everywhere to prevent some rare
# linker errors depending on what compiler is used
add_compile_options("-fPIC")

if(CMAKE_BUILD_TYPE MATCHES Debug)
	# leak detection doesn't work correctly when the code is called by
	# Python, so disable it
	if(DEFINED BUILD_PYTHON_MODULE)
		add_compile_options("-fno-omit-frame-pointer" "-fsanitize=address,undefined,bounds,pointer-overflow")
		add_link_options("-shared-libasan" "-fno-omit-frame-pointer" "-fsanitize=address,undefined,bounds,pointer-overflow")
	else()
		add_compile_options("-fno-omit-frame-pointer" "-fsanitize=address,undefined,bounds,pointer-overflow,leak")
		add_link_options("-fno-omit-frame-pointer" "-fsanitize=address,undefined,bounds,pointer-overflow,leak")
	endif()

	# mold is an extremely fast linker, use it if available
	# only use mold in debug mode, link time optimization currently doesn't
	# work with mold and provides large speedups
	find_program(MOLD_FOUND "mold")
	if(MOLD_FOUND)
		add_link_options("-fuse-ld=mold")
	endif()
else()
	add_compile_options("-flto" "-fno-math-errno")
	if (NOT DEFINED EMSCRIPTEN)
		# emscripten doesn't support -march=native, it doesn't make sense
		# for WASM anyway
		add_compile_options("-march=native")
	else()
		# tell emscripten to generate an HTML file that can be used to
		# test the WASM, and ensure necessary code is transformed to be
		# async friendly; it allows the game to be run much more smoothly
		set(CMAKE_EXECUTABLE_SUFFIX ".html")
		add_link_options("-sASYNCIFY")
	endif()
	# ensure the linker used is from the same compiler toolchain, or else
	# link time optimization will probably fail; if we're using
	# emscripten it will use it's own linker
	if(CMAKE_C_COMPILER_ID MATCHES "Clang" AND NOT DEFINED EMSCRIPTEN)
		add_link_options("-fuse-ld=lld")
	endif()
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(FETCHCONTENT_QUIET FALSE)

# fetch and configure dependencies
FetchContent_Declare(
	raylib
	URL https://github.com/raysan5/raylib/archive/c1ab645ca298a2801097931d1079b10ff7eb9df8.zip # 5.5
)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Statically link raylib" FORCE)
set(WITH_PIC "Compile static library as position-independent code" ON)
set(CUSTOMIZE_BUILD ON CACHE BOOL "Customize raylib build settings" FORCE)
set(USE_AUDIO OFF CACHE BOOL "Don't build unused audio module" FORCE)
FetchContent_MakeAvailable(raylib)

# if box2d is fetched first installing built python module will fail
# for reasons unbeknownst to mere mortals
# maybe due to install prefix schenanigans?
FetchContent_Declare(
	box2d
	URL https://github.com/capnspacehook/box2d/archive/7bdfaf243700aa4b2b82f5eb13f95359966fa5da.zip # 3.1
)
set(BOX2D_ENABLE_SIMD ON CACHE BOOL "Enable SIMD math (faster)" FORCE)
set(BOX2D_AVX2 ON CACHE BOOL "Enable AVX2 (faster)" FORCE)
add_compile_definitions(B2_MAX_WORLDS=65534)
FetchContent_MakeAvailable(box2d)
# this is set to off by box2d to enable cross platform determinism, but
# I don't care about that and want the small speedup instead
target_compile_options(box2d PRIVATE "-ffp-contract=fast")

function(configure_target target_name)
	target_include_directories(
		${target_name} PRIVATE
		"${CMAKE_CURRENT_SOURCE_DIR}/src"
		"${CMAKE_CURRENT_SOURCE_DIR}/src/include"
	)

	# Mark box2d as a system include directory to suppress warnings from it
	target_include_directories(${target_name} SYSTEM PRIVATE "${box2d_SOURCE_DIR}/src")

	target_link_libraries(${target_name} PRIVATE raylib box2d)

	target_compile_options(${target_name} PRIVATE
		"-Wall" "-Wextra" "-Wpedantic"
		"-Wno-implicit-fallthrough" "-Wno-variadic-macros" "-Wno-strict-prototypes" "-Wno-gnu-statement-expression"
	)
endfunction()

if(DEFINED BUILD_PYTHON_MODULE)
	# transpile Cython to C
	find_package(
		Python
		COMPONENTS Interpreter Development.Module
		REQUIRED
	)

	# add all project headers as a dependency to the autopxd command so
	# the PXD file will be regenerated if any of the headers change
	file(GLOB_RECURSE IMPULSE_WARS_HEADERS RELATIVE ${CMAKE_SOURCE_DIR} "src/*.h")

	add_custom_command(
		OUTPUT impulse_wars.pxd.raw
		COMMENT "Converting C headers to Cython PXD file"
		COMMAND autopxd
			-I "${CMAKE_CURRENT_SOURCE_DIR}/src"
			-I "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
			-I "${box2d_SOURCE_DIR}/include"
			-I "${raylib_SOURCE_DIR}/src"
			-D AUTOPXD
			"${CMAKE_CURRENT_SOURCE_DIR}/src/env.h" impulse_wars.pxd.raw
		DEPENDS "${IMPULSE_WARS_HEADERS}"
		VERBATIM
	)

	# workaround for autopxd not setting bools correctly
	add_custom_command(
		OUTPUT impulse_wars.pxd
		COMMENT "Fixing bools in Cython PXD file"
		COMMAND "${CMAKE_COMMAND}"
			"-DSOURCE=${CMAKE_CURRENT_BINARY_DIR}/impulse_wars.pxd.raw"
			"-DTARGET=${CMAKE_CURRENT_BINARY_DIR}/impulse_wars.pxd"
			-P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/fix-cython-pxd.cmake"
		DEPENDS impulse_wars.pxd.raw
		VERBATIM
	)

	# copy Cython file to binary dir to avoid "class is declared but not defined" errors
	# apparently Cython doesn't like when you aren't compiling a file in the working directory
	# and that file imports a PXD file
	configure_file(
		"${CMAKE_CURRENT_SOURCE_DIR}/cy_impulse_wars.pyx"
		"${CMAKE_CURRENT_BINARY_DIR}/cy_impulse_wars.pyx"
		COPYONLY
	)

	add_custom_command(
		OUTPUT impulse_wars.c
		COMMENT "Transpiling Cython to C"
		COMMAND Python::Interpreter -m cython
			-3
			-X boundscheck=false
			-X initializedcheck=false
			-X wraparound=false
			-X cdivision=true
			cy_impulse_wars.pyx -o impulse_wars.c
		DEPENDS impulse_wars.pxd cy_impulse_wars.pyx
		VERBATIM
	)

	# build python module
	python_add_library(cy_impulse_wars MODULE impulse_wars.c WITH_SOABI)

	configure_target(cy_impulse_wars)
	# disable false positive warnings for generated Cython source file
	# but keep other warnings enabled just in case something crazy is generated
	set_source_files_properties(impulse_wars.c PROPERTIES COMPILE_FLAGS "-Wno-pedantic")

	install(TARGETS cy_impulse_wars DESTINATION .)
elseif(DEFINED BUILD_DEMO)
	add_executable(demo "${CMAKE_CURRENT_SOURCE_DIR}/src/demo.c")
	configure_target(demo)
elseif(DEFINED BUILD_BENCHMARK)
	add_executable(benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark.c")
	configure_target(benchmark)
endif()
