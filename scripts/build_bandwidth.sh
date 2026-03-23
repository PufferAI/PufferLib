#!/bin/bash

# Usage: ./scripts/build_bandwidth.sh

FLAGS=(
    -Wall
    -I/usr/local/cuda/include
    "pufferlib/extensions/test_bandwidth.c"
    -o "test_bandwidth"
    -L/usr/local/cuda/lib64 -lcudart
    -lm
    -fPIC
)

clang -O2 -DNDEBUG ${FLAGS[@]}
