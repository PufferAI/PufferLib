#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VIEWER="$REPO_ROOT/build/fight_caves-viewer/fc_viewer"
PYTHON=${PYTHON:-python3}

cd "$REPO_ROOT"
"$PYTHON" ocean/fight_caves/scripts/preflight.py --mode viewer-runtime

if [ ! -x "$VIEWER" ]; then
    echo "Fight Caves viewer is not built: $VIEWER" >&2
    echo "Build it with: ./build.sh fight_caves --viewer" >&2
    exit 1
fi

exec "$VIEWER" "$@"
