#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SRC="$ROOT/ocean/affine_lock/tools/verify_transform_bruteforce.cu"
OUT="${TMPDIR:-/tmp}/affine_lock_verify_transform_bruteforce"
JSON_OUT="${TMPDIR:-/tmp}/affine_lock_verify_transform_bruteforce.json"
MITM_JSON_OUT="${TMPDIR:-/tmp}/affine_lock_verify_transform_mitm.json"

if [ ! -f "$SRC" ]; then
    echo "missing affine transform brute-force verifier: $SRC" >&2
    exit 1
fi

CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-}}"
if [ -z "$CUDA_HOME" ] && [ -f "$ROOT/.venv/bin/activate" ]; then
    # The project venv records CUDA_HOME on the machines used for GPU work.
    # Keep this local to the smoke script so run_all.sh stays self-contained.
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
    CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-}}"
fi
if [ -z "$CUDA_HOME" ] && command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
if [ -z "$CUDA_HOME" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "skipping affine transform brute-force smoke: nvcc not found"
    exit 0
fi

"$CUDA_HOME/bin/nvcc" \
    -std=c++17 -O2 \
    -I"$ROOT" -I"$ROOT/ocean/affine_lock" \
    "$SRC" -o "$OUT"

"$OUT" --cpu --max-depth 4 --write-json "$JSON_OUT"
"$OUT" --cpu --method mitm --max-depth 8 --write-json "$MITM_JSON_OUT"

python3 - "$JSON_OUT" "$MITM_JSON_OUT" <<'PY'
import json
import sys

raw_path, mitm_path = sys.argv[1], sys.argv[2]
with open(raw_path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
with open(mitm_path, "r", encoding="utf-8") as handle:
    mitm = json.load(handle)

expected_discovered = 1 + 8 + 25 + 54 + 104
expected_d8_discovered = expected_discovered + 192 + 346 + 610 + 1057
records = data["transforms"]
identity = next(
    record
    for record in records
    if record["perm_id"] == 0 and record["xor_mask"] == "0x0000"
)

assert "verified" not in data
assert data["scope_verified"] is True
assert data["verified_through_depth"] == 4
assert data["all_generated_transforms_verified"] is False
assert data["max_depth"] == 4
assert data["raw_action_strings_enumerated"] == 4681
assert data["distance_histogram_bruteforce"]["4"] == 104
assert len(records) == 16384
assert identity["generated_min_depth"] == 0
assert identity["bruteforce_min_depth"] == 0
assert identity["generated_action_ids"] == []
assert sum(1 for record in records
           if record["bruteforce_min_depth"] is not None) == expected_discovered

assert mitm["method"] == "mitm"
assert mitm["half_enumeration_mode"] == "cpu"
assert mitm["pair_composition_mode"] == "cpu"
assert mitm["max_depth"] == 8
assert mitm["distance_histogram_bruteforce"]["8"] == 1057
assert sum(1 for record in mitm["transforms"]
           if record["bruteforce_min_depth"] is not None) == (
               expected_d8_discovered
           )
PY
