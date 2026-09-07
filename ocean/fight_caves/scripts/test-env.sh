#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TEST_ROOT="$REPO_ROOT/build/fight_caves-tests"
MODE="${1:---core}"
PYTHON=${PYTHON:-python3}

case "$MODE" in
    --core|--puffer|--all) ;;
    *)
        echo "Usage: bash ocean/fight_caves/scripts/test-env.sh [--core|--puffer|--all]" >&2
        exit 2
        ;;
esac

cd "$REPO_ROOT"
"$PYTHON" -m pytest -q \
    tests/test_fight_caves_assets.py \
    tests/test_fight_caves_eval_contract.py \
    tests/test_fight_caves_layout.py
"$PYTHON" ocean/fight_caves/scripts/preflight.py --mode core

mkdir -p "$TEST_ROOT"
CORE_SOURCES=()
while IFS= read -r relative_source || [ -n "$relative_source" ]; do
    case "$relative_source" in
        ""|\#*) continue ;;
    esac
    CORE_SOURCES+=("ocean/fight_caves/$relative_source")
done < ocean/fight_caves/sources.txt

"${CC:-clang}" -std=c11 -O2 -Wall -Wextra -Werror \
    -Iocean/fight_caves/include -Iocean/fight_caves/src \
    ocean/fight_caves/tests/core_contract_test.c "${CORE_SOURCES[@]}" \
    -lm -o "$TEST_ROOT/core_contract_test"
"$TEST_ROOT/core_contract_test"

if [ "$MODE" = "--puffer" ] || [ "$MODE" = "--all" ]; then
    "$PYTHON" ocean/fight_caves/scripts/preflight.py --mode cpu
    ./build.sh fight_caves --cpu
    "$PYTHON" ocean/fight_caves/tests/puffer_contract_test.py
fi

if [ "$MODE" = "--all" ]; then
    "$PYTHON" ocean/fight_caves/scripts/preflight.py --mode viewer
    ./build.sh fight_caves --viewer
fi

echo "Fight Caves environment tests passed ($MODE)."
