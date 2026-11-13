#!/usr/bin/env bash
# Clones FinlaySanders/fast-nle into vendor/ and builds libnethack.so. Run by
# build.sh from config/nethack.ini [build]; web.sh has the wasm variant.
set -e
NLE_DIR="$PUFFER_ROOT/vendor/fast-nle"
if [ ! -d "$NLE_DIR/src" ]; then
    echo "Cloning fast-nle ..."
    git clone --depth 1 https://github.com/FinlaySanders/fast-nle.git "$NLE_DIR"
fi
if [ ! -f "$NLE_DIR/build/libnethack.so" ]; then
    echo "Building libnethack.so ..."
    cmake -S "$NLE_DIR" -B "$NLE_DIR/build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$NLE_DIR/build" --target nethack -j"$(getconf _NPROCESSORS_ONLN)"
fi
