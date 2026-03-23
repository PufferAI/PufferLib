#!/bin/bash

# Usage: ./build_vec.sh <env_name>

ENV_NAME=${1:?Usage: ./build_vec.sh <env_name>}

RAYLIB_NAME='raylib-5.5_linux_amd64'
LINK_ARCHIVES="./$RAYLIB_NAME/lib/libraylib.a"

FLAGS=(
    -shared
    -Wall
    -I./$RAYLIB_NAME/include
    -I/usr/local/cuda/include
    "pufferlib/extensions/${ENV_NAME}.c" -o "${ENV_NAME}.so"
    $LINK_ARCHIVES
    -lGL
    -lm
    -lpthread
    -fopenmp
    #-L/usr/local/cuda/lib64 -lcudart
    -ferror-limit=3
    -DPLATFORM_DESKTOP
    # Bite me
    -Werror=incompatible-pointer-types
    -Wno-error=incompatible-pointer-types-discards-qualifiers
    -Wno-incompatible-pointer-types-discards-qualifiers
    -Wno-error=array-parameter
    -fms-extensions
    -fno-semantic-interposition
    #-fsanitize=address,undefined,bounds,pointer-overflow,leak
    #-fno-omit-frame-pointer
    #-fsanitize=thread
    -fPIC
    -fvisibility=hidden
)

clang -O2 -DNDEBUG ${FLAGS[@]}
