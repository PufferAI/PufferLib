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

	# add_compile_options("-pg")
	# add_link_options("-pg")
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
	URL https://github.com/capnspacehook/box2d/archive/df25d747be0ab2fd9425eece022d2ec897c2028d.zip
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
		"${CMAKE_CURRENT_SOURCE_DIR}"
		"${CMAKE_CURRENT_SOURCE_DIR}/include"
	)

	# Mark box2d as a system include directory to suppress warnings from it
	target_include_directories(${target_name} SYSTEM PRIVATE "${box2d_SOURCE_DIR}/src")

	target_link_libraries(${target_name} PRIVATE raylib box2d)

	target_compile_options(${target_name} PRIVATE
		"-Werror" "-Wall" "-Wextra" "-Wpedantic"
		"-Wno-implicit-fallthrough" "-Wno-variadic-macros" "-Wno-strict-prototypes" "-Wno-gnu-statement-expression"
	)
endfunction()

if(DEFINED BUILD_PYTHON_MODULE)
	find_package(
		Python
		COMPONENTS Interpreter Development.Module NumPy
		REQUIRED
	)

	python_add_library(binding MODULE binding.c WITH_SOABI)

	target_include_directories(binding PRIVATE
    	${Python_NumPy_INCLUDE_DIRS}
	)

	configure_target(binding)

	install(TARGETS binding DESTINATION .)
elseif(DEFINED BUILD_DEMO)
	add_executable(demo "${CMAKE_CURRENT_SOURCE_DIR}/impulse_wars.c")
	configure_target(demo)
elseif(DEFINED BUILD_BENCHMARK)
	add_executable(benchmark "${CMAKE_CURRENT_SOURCE_DIR}/benchmark.c")
	configure_target(benchmark)
endif()
