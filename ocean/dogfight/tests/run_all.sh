#!/usr/bin/env bash
#
# run_all.sh - Compile and run all dogfight regression tests (C + Python).
#
# Usage (from the repo root):
#   bash ocean/dogfight/tests/run_all.sh
#
# Exits 0 if every test binary/script exits 0, non-zero otherwise. Stdout
# is terse; per-file output (including failed-assertion lines on stderr)
# is preserved so CI logs show which test and which line failed.
#
# Picks up:
#   test_*.c   compiled with raylib + libm and executed
#   test_*.py  executed with .venv/bin/python (skipped if missing)

set -u

# Resolve repo root (directory containing this script is ocean/dogfight/tests)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RAYLIB_INC="raylib-5.5_linux_amd64/include"
RAYLIB_LIB="raylib-5.5_linux_amd64/lib/libraylib.a"
DOGFIGHT_INC="ocean/dogfight"

if [[ ! -f "${RAYLIB_LIB}" ]]; then
    echo "ERROR: raylib static lib not found at ${RAYLIB_LIB}" >&2
    echo "       run ./build.sh once to populate raylib-5.5_linux_amd64/" >&2
    exit 2
fi

CC="${CC:-gcc}"
CFLAGS="-O2 -Wall -I ${DOGFIGHT_INC} -I ${RAYLIB_INC}"
LDFLAGS="${RAYLIB_LIB} -lm -lpthread -ldl"

n_total=0
n_pass=0
n_fail=0
failing=()

shopt -s nullglob
for src in "${SCRIPT_DIR}"/test_*.c; do
    name="$(basename "${src}" .c)"
    bin="${SCRIPT_DIR}/${name}"
    n_total=$((n_total + 1))

    if ! ${CC} ${CFLAGS} "${src}" ${LDFLAGS} -o "${bin}" 2>&1; then
        echo "[${name}] COMPILE FAILED"
        n_fail=$((n_fail + 1))
        failing+=("${name}")
        continue
    fi

    if "${bin}"; then
        n_pass=$((n_pass + 1))
    else
        n_fail=$((n_fail + 1))
        failing+=("${name}")
    fi
done

# Python tests
PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
    PY="$(command -v python3 || command -v python || true)"
fi
for src in "${SCRIPT_DIR}"/test_*.py; do
    name="$(basename "${src}" .py)"
    n_total=$((n_total + 1))
    if [[ -z "${PY}" ]]; then
        echo "[${name}] SKIPPED (no python interpreter found)"
        n_fail=$((n_fail + 1))
        failing+=("${name}")
        continue
    fi
    if "${PY}" "${src}"; then
        n_pass=$((n_pass + 1))
    else
        n_fail=$((n_fail + 1))
        failing+=("${name}")
    fi
done

echo
echo "=========================================="
echo "  Dogfight regression tests (C + Python)"
echo "=========================================="
echo "  total:  ${n_total}"
echo "  passed: ${n_pass}"
echo "  failed: ${n_fail}"
if [[ ${n_fail} -gt 0 ]]; then
    echo "  failing binaries:"
    for name in "${failing[@]}"; do
        echo "    - ${name}"
    done
fi
echo "=========================================="

exit "${n_fail}"
