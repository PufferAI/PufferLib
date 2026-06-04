#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

mkdir -p build/pathfinder-tests

${CC:-clang} -std=c11 -Wall -Wextra -Werror \
    -I. -Iocean/pathfinder -Ivendor -Iraylib-5.5_linux_amd64/include \
    ocean/pathfinder/tests/test_pathfinder_core.c \
    -lm \
    -o build/pathfinder-tests/test_pathfinder_core

build/pathfinder-tests/test_pathfinder_core
