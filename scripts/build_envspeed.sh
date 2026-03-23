#!/bin/bash

# Usage: ./scripts/build_envspeed.sh <env_name>
# Example: ./scripts/build_envspeed.sh breakout

if [ -z "$1" ]; then
    echo "Usage: $0 <env_name>"
    echo "Example: $0 breakout"
    exit 1
fi

ENV_NAME=$1
STATIC_LIB="pufferlib/extensions/libstatic_${ENV_NAME}.a"

if [ ! -f "$STATIC_LIB" ]; then
    echo "Error: Static library not found: $STATIC_LIB"
    echo "Build it first with: ./scripts/build_static_${ENV_NAME}.sh"
    exit 1
fi

RAYLIB_NAME='raylib-5.5_linux_amd64'
LINK_ARCHIVES="./$RAYLIB_NAME/lib/libraylib.a"

FLAGS=(
    -Wall
    -I./$RAYLIB_NAME/include
    -I/usr/local/cuda/include
    -Ipufferlib/extensions
    "pufferlib/extensions/test_envspeed_static.c"
    "pufferlib/extensions/ini.c"
    "$STATIC_LIB"
    -o "test_envspeed_static"
    $LINK_ARCHIVES
    -lGL
    -lm
    -lpthread
    -L/usr/local/cuda/lib64 -lcudart
    -ferror-limit=3
    -DPLATFORM_DESKTOP
    # Bite me
    -Werror=incompatible-pointer-types
    -Wno-error=incompatible-pointer-types-discards-qualifiers
    -Wno-incompatible-pointer-types-discards-qualifiers
    -Wno-error=array-parameter
    -fms-extensions
    #-fsanitize=address,undefined,bounds,pointer-overflow,leak
    #-fno-omit-frame-pointer
    #-fsanitize=thread
    -fPIC
)

clang -O2 -DNDEBUG -fopenmp -DENV_NAME=\"${ENV_NAME}\" ${FLAGS[@]}
