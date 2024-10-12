#!/bin/bash

# Usage: ./build_env.sh pong [local|web]

ENV=$1
MODE=${2:-local}

SRC_DIR="pufferlib/environments/ocean/$ENV"
OUTPUT_DIR="$SRC_DIR"
WEB_OUTPUT_DIR="build_web/$ENV/"
RESOURCES_DIR="$SRC_DIR/resources"

# Create build output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$WEB_OUTPUT_DIR"

if [ "$MODE" = "local" ]; then
    echo "Building $ENV for local testing..."
    gcc -g -O2 -Wall -I./raylib-5.0_linux_amd64/include \
        "$SRC_DIR/$ENV.c" -o "$OUTPUT_DIR/$ENV" \
        ./raylib-5.0_linux_amd64/lib/libraylib.a -lm -lpthread

    echo "Build completed: $OUTPUT_DIR/$ENV"
elif [ "$MODE" = "web" ]; then
    echo "Building $ENV for web deployment..."

    PRELOAD=""
    if [ -d "$RESOURCES_DIR" ]; then
        PRELOAD="--preload-file $RESOURCES_DIR"
    fi

    emcc \
        -o "$WEB_OUTPUT_DIR/$ENV.html" \
        "$SRC_DIR/$ENV.c" \
        -Os \
        -Wall \
        ./raylib-5.0_webassembly/lib/libraylib.a \
        -I./raylib-5.0/src \
        -L. \
        -L./raylib-5.0_webassembly/lib/libraylib.a \
        -sASSERTIONS=2 \
        -gsource-map \
        -s USE_GLFW=3 \
        -s USE_WEBGL2=1 \
        -s ASYNCIFY \
        -sFILESYSTEM \
        -s FORCE_FILESYSTEM=1 \
        --shell-file ./raylib-5.0/src/minshell.html \
        -DPLATFORM_WEB \
        -DGRAPHICS_API_OPENGL_ES3 \ 
        $PRELOAD
    echo "Web build completed: $OUTPUT_DIR/$ENV.html"
else
    echo "Invalid mode specified. Use 'local' or 'web'."
    exit 1
fi

