#!/bin/bash

# Usage: ./build_env.sh pong [local|fast|web]

RAYLIB_NAME='raylib-5.5_linux_amd64'
LINK_ARCHIVES="./$RAYLIB_NAME/lib/libraylib.a"

FLAGS=(
    -Wall
    -I./$RAYLIB_NAME/include
    -I/usr/local/cuda/include
    "pufferlib/extensions/test_dll.c" -o "test_dll"
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
    #-fsanitize=address,undefined,bounds,pointer-overflow,leak
    #-fsanitize=thread
    -fno-omit-frame-pointer
)

clang -g -O2 ${FLAGS[@]}
