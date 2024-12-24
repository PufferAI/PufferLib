#!/bin/bash

# Usage: ./build_env.sh pong [local|fast|web]

ENV=$1
MODE=${2:-local}
PLATFORM="$(uname -s)"
SRC_DIR="pufferlib/ocean/$ENV"
WEB_OUTPUT_DIR="build_web/$ENV"

# Create build output directory
mkdir -p "$WEB_OUTPUT_DIR"

if [ "$MODE" = "web" ]; then
    echo "Building $ENV for web deployment..."
    emcc \
        -o "$WEB_OUTPUT_DIR/game.html" \
        "$SRC_DIR/$ENV.c" \
        -O3 \
        -Wall \
        ./raylib_wasm/lib/libraylib.a \
        -I./raylib_wasm/include \
        -I./pufferlib\
        -L. \
        -L./raylib_wasm/lib \
        -sASSERTIONS=2 \
        -gsource-map \
        -s USE_GLFW=3 \
        -s USE_WEBGL2=1 \
        -s ASYNCIFY \
        -sFILESYSTEM \
        -s FORCE_FILESYSTEM=1 \
        --shell-file ./scripts/minshell.html \
        -sINITIAL_MEMORY=512MB \
        -sSTACK_SIZE=512KB \
        -DPLATFORM_WEB \
        -DGRAPHICS_API_OPENGL_ES3 \
        --preload-file pufferlib/resources@resources/
    echo "Web build completed: $WEB_OUTPUT_DIR/game.html"
    exit 0
fi

FLAGS=(
    -Wall
    -I./raylib/include
    -I./pufferlib
    "$SRC_DIR/$ENV.c" -o "$ENV"
    ./raylib/lib/libraylib.a
    -lm
    -lpthread
    -DPLATFORM_DESKTOP
)


if [ "$PLATFORM" = "Darwin" ]; then
    FLAGS+=(
        -framework Cocoa
        -framework IOKit
        -framework CoreVideo
    )
fi

echo ${FLAGS[@]}

if [ "$MODE" = "local" ]; then
    echo "Building $ENV for local testing..."
    if [ "$PLATFORM" = "Linux" ]; then
        # These important debug flags don't work on macos
        FLAGS+=(
            -fsanitize=address,undefined,bounds,pointer-overflow,leak
        )
    fi
    clang -g -O0 ${FLAGS[@]}
elif [ "$MODE" = "fast" ]; then
    echo "Building optimized $ENV for local testing..."
    clang -pg -O2 ${FLAGS[@]}
    echo "Built to: $ENV"
else
    echo "Invalid mode specified: local|fast|web"
    exit 1
fi
