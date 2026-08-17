#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
OUT="${TMPDIR:-/tmp}/affine_lock_tests"
LOG_OUT="${TMPDIR:-/tmp}/affine_lock_log_export_tests"
C99_OUT="${TMPDIR:-/tmp}/affine_lock_c99_compile"
CC_BIN="${CC:-clang}"
RAYLIB_INC="$ROOT/raylib-5.5_linux_amd64/include"
RAYLIB_LIB="$ROOT/raylib-5.5_linux_amd64/lib/libraylib.a"
if [ ! -d "$RAYLIB_INC" ]; then
    RAYLIB_INC="$ROOT/raylib-5.5_macos/include"
    RAYLIB_LIB="$ROOT/raylib-5.5_macos/lib/libraylib.a"
fi
if [ "$(uname -s)" = "Linux" ]; then
    RAYLIB_LDFLAGS=(-lGL -lpthread -ldl -lrt)
else
    RAYLIB_LDFLAGS=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
fi

python3 "$ROOT/ocean/affine_lock/tests/test_metadata_smoke.py"
bash "$ROOT/ocean/affine_lock/tests/test_8action_visible_targets_smoke.sh"

"$CC_BIN" \
  -std=c99 -pedantic -Wall -Wextra -Werror -Wno-unused-function \
  -D_POSIX_C_SOURCE=200809L \
  -O0 -ffunction-sections -fdata-sections \
  -I"$ROOT" -I"$ROOT/src" -I"$ROOT/ocean/affine_lock" -I"$ROOT/vendor" \
  -I"$RAYLIB_INC" \
  "$ROOT/ocean/affine_lock/tests/test_affine_lock.c" \
  "$RAYLIB_LIB" -Wl,--gc-sections "${RAYLIB_LDFLAGS[@]}" -lm -o "$C99_OUT"

"$CC_BIN" \
  -std=c11 -Wall -Wextra -Werror -Wno-unused-function \
  -D_POSIX_C_SOURCE=200809L \
  -O0 -g -ffunction-sections -fdata-sections -fsanitize=address,undefined \
  -I"$ROOT" -I"$ROOT/src" -I"$ROOT/ocean/affine_lock" -I"$ROOT/vendor" \
  -I"$RAYLIB_INC" \
  "$ROOT/ocean/affine_lock/tests/test_affine_lock.c" \
  "$RAYLIB_LIB" -Wl,--gc-sections "${RAYLIB_LDFLAGS[@]}" -lm -o "$OUT"

"$CC_BIN" \
  -std=c11 -Wall -Wextra -Werror -Wno-unused-function -Wno-unused-parameter \
  -D_POSIX_C_SOURCE=200809L \
  -O0 -g -ffunction-sections -fdata-sections -fsanitize=address,undefined \
  -I"$ROOT" -I"$ROOT/src" -I"$ROOT/ocean/affine_lock" -I"$ROOT/vendor" \
  -I"$RAYLIB_INC" \
  "$ROOT/ocean/affine_lock/tests/test_affine_lock_log_export.c" \
  "$RAYLIB_LIB" -Wl,--gc-sections "${RAYLIB_LDFLAGS[@]}" -lm -o "$LOG_OUT"

ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}" "$OUT"
ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}" "$LOG_OUT"
