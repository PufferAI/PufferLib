#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VALIDATION_ROOT="$REPO_ROOT/build/fight_caves-acceptance"
PYTHON=${PYTHON:-python3}

fail() {
    echo "Fight Caves checkout validation failed: $*" >&2
    exit 1
}

expect_failure() {
    local description=$1
    local expected=$2
    shift 2
    local log="$VALIDATION_ROOT/expected-failure.log"
    if "$@" >"$log" 2>&1; then
        fail "$description unexpectedly succeeded"
    fi
    if ! grep -F "$expected" "$log" >/dev/null; then
        echo "Expected failure output:" >&2
        sed -n '1,120p' "$log" >&2
        fail "$description did not explain how it failed"
    fi
    echo "Expected failure passed: $description"
}

cd "$REPO_ROOT"
mkdir -p "$VALIDATION_ROOT"
RUN_ROOT=$(mktemp -d "$VALIDATION_ROOT/run.XXXXXX")

for variable in FC_ASSET_ROOT FC_REPO_ROOT FC_COLLISION_PATH FC_MOVEMENT_PATH FC_LOS_PATH FC_COMPILED_BACKEND_PATH FC_CHECKPOINT_ROOT; do
    unset "$variable" || true
done

expect_failure \
    "missing asset bundle" \
    "Install assets with: bash ocean/fight_caves/scripts/setup-data.sh --all" \
    "$PYTHON" ocean/fight_caves/scripts/preflight.py --mode core

bash ocean/fight_caves/scripts/setup-data.sh --all
bash ocean/fight_caves/scripts/setup-data.sh --all --verify-only

bash ocean/fight_caves/scripts/test-env.sh --all
./build.sh fight_caves --fast
./fight_caves >"$VALIDATION_ROOT/native-smoke.log"
grep -F "Episodes:    100" "$VALIDATION_ROOT/native-smoke.log" >/dev/null \
    || fail "standalone environment did not finish its smoke run"
mv fight_caves "$VALIDATION_ROOT/fight_caves"

CORE_MAP="resources/fight_caves/runtime/fightcaves.collision"
mv "$CORE_MAP" "$VALIDATION_ROOT/fightcaves.collision"
expect_failure \
    "missing core map at runtime" \
    "required Fight Caves arena asset 'fightcaves.collision' is missing" \
    build/fight_caves-tests/core_contract_test
mv "$VALIDATION_ROOT/fightcaves.collision" "$CORE_MAP"

expect_failure \
    "invalid explicit core map override" \
    "FC_COLLISION_PATH points to an unreadable" \
    env FC_COLLISION_PATH="$VALIDATION_ROOT/not-a-map" \
    build/fight_caves-tests/core_contract_test

VIEWER_ASSET="resources/fight_caves/viewer/fightcaves.minimap.png"
mv "$VIEWER_ASSET" "$VALIDATION_ROOT/fightcaves.minimap.png"
expect_failure \
    "missing viewer asset" \
    "viewer asset bundle is invalid: missing viewer/fightcaves.minimap.png" \
    bash ocean/fight_caves/scripts/run-viewer.sh --screenshot "$VALIDATION_ROOT/missing.png"
mv "$VALIDATION_ROOT/fightcaves.minimap.png" "$VIEWER_ASSET"

"$PYTHON" -m pufferlib.pufferl train fight_caves \
    --slowly \
    --train.gpus 1 \
    --train.total-timesteps 4096 \
    --train.horizon 32 \
    --train.minibatch-size 512 \
    --train.replay-ratio 0.25 \
    --vec.total-agents 64 \
    --vec.num-buffers 1 \
    --vec.num-threads 1 \
    --checkpoint-dir "$RUN_ROOT/checkpoints" \
    --log-dir "$RUN_ROOT/logs" \
    --checkpoint-interval 1000000 \
    >"$VALIDATION_ROOT/training-smoke.log" 2>&1

CHECKPOINT=$(FC_VALIDATION_RUN_ROOT="$RUN_ROOT" "$PYTHON" - <<'PY'
import os
from pathlib import Path
paths = list(
    (Path(os.environ["FC_VALIDATION_RUN_ROOT"]) / "checkpoints" / "fight_caves")
    .glob("**/*.bin")
)
if not paths:
    raise SystemExit("training smoke did not create a checkpoint")
print(max(paths, key=lambda path: path.stat().st_mtime))
PY
)

cp "$CHECKPOINT" "$VALIDATION_ROOT/wrong-size.bin"
truncate -s 64 "$VALIDATION_ROOT/wrong-size.bin"
expect_failure \
    "incompatible checkpoint" \
    "checkpoint rejected" \
    "$PYTHON" ocean/fight_caves/viewer/eval_viewer.py \
        --ckpt "$VALIDATION_ROOT/wrong-size.bin" --max-ticks 1

if command -v xvfb-run >/dev/null 2>&1; then
    DISPLAY_PREFIX=(xvfb-run -a)
elif [ -n "${DISPLAY:-}" ]; then
    DISPLAY_PREFIX=()
else
    fail "viewer validation needs xvfb-run or an existing DISPLAY"
fi

SCREENSHOT_NAME="fight-caves-acceptance.png"
"${DISPLAY_PREFIX[@]}" bash ocean/fight_caves/scripts/run-viewer.sh \
    --screenshot "$SCREENSHOT_NAME" \
    >"$VALIDATION_ROOT/viewer-smoke.log" 2>&1
test -s "$SCREENSHOT_NAME" \
    || fail "playable viewer did not create a screenshot"
mv "$SCREENSHOT_NAME" "$VALIDATION_ROOT/playable.png"

"${DISPLAY_PREFIX[@]}" "$PYTHON" ocean/fight_caves/viewer/eval_viewer.py \
    --ckpt "$CHECKPOINT" --speed 10 --max-ticks 25 \
    >"$VALIDATION_ROOT/replay-smoke.log" 2>&1
grep -F "[eval] Policy ready (CPU)" "$VALIDATION_ROOT/replay-smoke.log" >/dev/null \
    || fail "checkpoint replay did not load the policy"
grep -F "[eval] Smoke limit reached at tick 25" "$VALIDATION_ROOT/replay-smoke.log" >/dev/null \
    || fail "checkpoint replay did not advance through the viewer"

bash ocean/fight_caves/scripts/setup-data.sh --all --verify-only
echo "Fight Caves clean-checkout validation passed."
