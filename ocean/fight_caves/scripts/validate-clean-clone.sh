#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SOURCE=${FC_CLEAN_CLONE_SOURCE:-$(git -C "$REPO_ROOT" remote get-url origin)}
REF=${FC_CLEAN_CLONE_REF:-$(git -C "$REPO_ROOT" branch --show-current)}
KEEP=${FC_CLEAN_CLONE_KEEP:-0}
SYSTEM_SITE_PACKAGES=${FC_CLEAN_CLONE_SYSTEM_SITE_PACKAGES:-0}
SKIP_PIP=${FC_CLEAN_CLONE_SKIP_PIP:-0}

TEMP_ROOT=$(mktemp -d -t fight-caves-clean-clone-XXXXXX)
cleanup() {
    if [ "$KEEP" = "1" ]; then
        echo "Kept clean-clone workspace: $TEMP_ROOT"
    else
        rm -rf "$TEMP_ROOT"
    fi
}
trap cleanup EXIT

echo "Cloning $SOURCE at $REF into isolated workspace"
git clone --quiet --branch "$REF" --single-branch "$SOURCE" "$TEMP_ROOT/PufferLib"
cd "$TEMP_ROOT/PufferLib"

test -z "$(git status --porcelain)" \
    || { echo "Fresh checkout is unexpectedly dirty" >&2; exit 1; }
test ! -e resources/fight_caves/runtime
test ! -e resources/fight_caves/viewer

VENV_ARGS=()
if [ "$SYSTEM_SITE_PACKAGES" = "1" ]; then
    VENV_ARGS+=(--system-site-packages)
fi
python3 -m venv "${VENV_ARGS[@]}" .venv
if [ "$SKIP_PIP" != "1" ]; then
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    .venv/bin/python -m pip install -e . --no-build-isolation
    .venv/bin/python -m pip install pytest
fi

export PYTHON="$PWD/.venv/bin/python"
bash ocean/fight_caves/scripts/validate-checkout.sh

test -z "$(git status --porcelain)" \
    || { echo "Validation changed tracked files in the clean clone" >&2; git status --short >&2; exit 1; }
echo "Fight Caves isolated clean-clone validation passed."
