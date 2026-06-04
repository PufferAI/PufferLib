#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

mkdir -p build/pathfinder-tests

${CC:-clang} -std=c11 -O3 -DNDEBUG -Wall -Wextra -Werror \
    -I. -Iocean/pathfinder -Ivendor -Iraylib-5.5_linux_amd64/include \
    ocean/pathfinder/tests/bench_pathfinder.c \
    -lm \
    -o build/pathfinder-tests/bench_pathfinder

build/pathfinder-tests/bench_pathfinder "$@"

if [[ "${PATHFINDER_TRAIN_BENCH:-0}" != "0" ]]; then
    source .venv/bin/activate
    ./build.sh pathfinder
    python ocean/pathfinder/tests/benchmark_pathfinder.py \
        --timesteps "${PATHFINDER_TRAIN_TIMESTEPS:-2097152}" \
        ${PATHFINDER_TRAIN_BENCH_ARGS:-}
fi
