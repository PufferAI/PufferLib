#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

mkdir -p build/bat-tests
cc -std=c99 -O2 -Wall -Wextra -DBAT_HEADLESS \
    -I. -Iocean/bat -Ivendor -Iraylib-5.5_linux_amd64/include \
    ocean/bat/tests/test_bat_core.c \
    -lm \
    -o build/bat-tests/test_bat_core

build/bat-tests/test_bat_core
