#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="${TMPDIR:-/tmp}/affine_lock_tests"
LOG_OUT="${TMPDIR:-/tmp}/affine_lock_log_export_tests"
CC_BIN="${CC:-clang}"

python3 "$ROOT/ocean/affine_lock/tools/generate_transform_table.py" --check
python3 "$ROOT/ocean/affine_lock/tools/replay_benchmarks.py" --check
python3 "$ROOT/ocean/affine_lock/tests/test_metadata_smoke.py"
bash "$ROOT/ocean/affine_lock/tests/test_transform_bruteforce_smoke.sh"
bash "$ROOT/ocean/affine_lock/tests/test_odd7_visible_targets_smoke.sh"

"$CC_BIN" \
  -std=c11 -Wall -Wextra -Werror -Wno-unused-function \
  -O0 -g -fsanitize=address,undefined \
  -I"$ROOT" -I"$ROOT/src" -I"$ROOT/ocean/affine_lock" -I"$ROOT/vendor" \
  "$ROOT/ocean/affine_lock/tests/test_affine_lock.c" \
  -lm -o "$OUT"

"$CC_BIN" \
  -std=c11 -Wall -Wextra -Werror -Wno-unused-function -Wno-unused-parameter \
  -D_POSIX_C_SOURCE=200809L \
  -O0 -g -ffunction-sections -fdata-sections -fsanitize=address,undefined \
  -I"$ROOT" -I"$ROOT/src" -I"$ROOT/ocean/affine_lock" -I"$ROOT/vendor" \
  "$ROOT/ocean/affine_lock/tests/test_affine_lock_log_export.c" \
  -Wl,--gc-sections -lm -o "$LOG_OUT"

ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}" "$OUT"
ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}" "$LOG_OUT"
