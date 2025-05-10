#!/bin/bash

# Usage: ./build.sh your_file.c [debug|release]

SOURCE=$1
MODE=${2:-debug}
PLATFORM="$(uname -s)"

# Extract filename without extension
FILENAME=$(basename -- "$SOURCE")
FILENAME="${FILENAME%.*}"

FLAGS=(
    -Wall
    "$SOURCE" -o "$FILENAME"
    -lm
    -lpthread
)

if [ "$PLATFORM" = "Darwin" ]; then
    FLAGS+=(
        -framework Cocoa
        -framework IOKit
        -framework CoreVideo
    )
fi

echo "Compiling with: ${FLAGS[@]}"

if [ "$MODE" = "debug" ]; then
    echo "Building $SOURCE in debug mode..."
    if [ "$PLATFORM" = "Linux" ]; then
        # These important debug flags don't work on macos
        FLAGS+=(
            -fsanitize=address,undefined,bounds,pointer-overflow,leak -g
        )
    fi  
    clang -g -O0 ${FLAGS[@]}
    echo "Built to: $FILENAME (debug mode)"
elif [ "$MODE" = "release" ]; then
    echo "Building optimized $SOURCE..."
    clang -O2 ${FLAGS[@]}
    echo "Built to: $FILENAME (release mode)"
else
    echo "Invalid mode specified: debug|release"
    exit 1
fi