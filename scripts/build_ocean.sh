#!/bin/bash

# Usage: ./build_env.sh pong [local|fast|web]

ENV=$1
MODE=${2:-local}
PLATFORM="$(uname -s)"
SRC_DIR="pufferlib/ocean/$ENV"
WEB_OUTPUT_DIR="build_web/$ENV"
RAYLIB_NAME='raylib-5.5_macos'
BOX2D_NAME='box2d-macos-arm64'
if [ "$PLATFORM" = "Linux" ]; then
    RAYLIB_NAME='raylib-5.5_linux_amd64'
    BOX2D_NAME='box2d-linux-amd64'
fi
if [ "$MODE" = "web" ]; then
    RAYLIB_NAME='raylib-5.5_webassembly'
    BOX2D_NAME='box2d-web'
fi

LINK_ARCHIVES="./$RAYLIB_NAME/lib/libraylib.a"
if [ "$ENV" = "impulse_wars" ]; then
    LINK_ARCHIVES="$LINK_ARCHIVES ./$BOX2D_NAME/libbox2d.a"
fi

# Create build output directory
mkdir -p "$WEB_OUTPUT_DIR"

if [ "$MODE" = "web" ]; then
    echo "Building $ENV for web deployment..."
    emcc \
        -o "$WEB_OUTPUT_DIR/game.html" \
        "$SRC_DIR/$ENV.c" \
        -O3 \
        -Wall \
        $LINK_ARCHIVES \
        -I./$RAYLIB_NAME/include \
        -I./$BOX2D_NAME/include \
        -I./$BOX2D_NAME/src \
        -I./pufferlib/extensions \
        -I./pufferlib \
        -L. \
        -L./$RAYLIB_NAME/lib \
        -sASSERTIONS=2 \
        -gsource-map \
        -s USE_GLFW=3 \
        -s USE_WEBGL2=1 \
        -s ASYNCIFY \
        -sFILESYSTEM \
        -s FORCE_FILESYSTEM=1 \
        --shell-file ./scripts/minshell.html \
        -sINITIAL_MEMORY=512MB \
        -sALLOW_MEMORY_GROWTH \
        -sSTACK_SIZE=512KB \
        -DNDEBUG \
        -DPLATFORM_WEB \
        -DGRAPHICS_API_OPENGL_ES3 \
        --preload-file pufferlib/resources/$1@resources/$1 \
        --preload-file pufferlib/resources/shared@resources/shared 
    echo "Web build completed: $WEB_OUTPUT_DIR/game.html"
    echo "Preloaded files:"
    echo "  pufferlib/resources/$1@resources$1"
    echo "  pufferlib/resources/shared@resources/shared"
    exit 0
fi

FLAGS=(
    -Wall
    -I./$RAYLIB_NAME/include
    -I./$BOX2D_NAME/include
    -I./$BOX2D_NAME/src
    -I./pufferlib/extensions
    "$SRC_DIR/$ENV.c" -o "$ENV"
    $LINK_ARCHIVES
    -lm
    -lpthread
    -ferror-limit=3
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
            -fno-omit-frame-pointer
        )
    fi  
    clang -g -O0 ${FLAGS[@]}
elif [ "$MODE" = "fast" ]; then
    echo "Building optimized $ENV for local testing..."
    clang -pg -O2 -DNDEBUG ${FLAGS[@]}
    echo "Built to: $ENV"
else
    echo "Invalid mode specified: local|fast|web"
    exit 1
fi
