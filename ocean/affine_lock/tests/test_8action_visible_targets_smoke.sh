#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SRC="$ROOT/ocean/affine_lock/tools/generate_8action_visible_targets.c"
OUT="${TMPDIR:-/tmp}/affine_lock_generate_8action_visible_targets"
LOADER_SRC="$ROOT/ocean/affine_lock/tests/test_visible_targets_loader.c"
LOADER_OUT="${TMPDIR:-/tmp}/affine_lock_test_visible_targets_loader"
BIN_OUT="${TMPDIR:-/tmp}/affine_lock_8action_visible_targets.bin"
JSON_OUT="${TMPDIR:-/tmp}/affine_lock_8action_visible_targets.json"
SEED_42_A_BIN_OUT="${TMPDIR:-/tmp}/affine_lock_8action_visible_targets_seed42_a.bin"
SEED_42_A_JSON_OUT="${TMPDIR:-/tmp}/affine_lock_8action_visible_targets_seed42_a.json"
SEED_42_B_BIN_OUT="${TMPDIR:-/tmp}/affine_lock_8action_visible_targets_seed42_b.bin"
SEED_42_B_JSON_OUT="${TMPDIR:-/tmp}/affine_lock_8action_visible_targets_seed42_b.json"
SEED_69_BIN_OUT="${TMPDIR:-/tmp}/affine_lock_8action_visible_targets_seed69.bin"
SEED_69_JSON_OUT="${TMPDIR:-/tmp}/affine_lock_8action_visible_targets_seed69.json"
FOUR_BIN_OUT="${TMPDIR:-/tmp}/affine_lock_4action_visible_targets.bin"
FOUR_JSON_OUT="${TMPDIR:-/tmp}/affine_lock_4action_visible_targets.json"
CC_BIN="${CC:-gcc}"

if [ ! -f "$SRC" ]; then
    echo "missing 8-action visible target generator: $SRC" >&2
    exit 1
fi
if [ ! -f "$LOADER_SRC" ]; then
    echo "missing visible target loader test: $LOADER_SRC" >&2
    exit 1
fi

"$CC_BIN" \
    -std=c11 -O3 -DNDEBUG -fopenmp \
    -Wall -Wextra -Werror \
    -I"$ROOT" -I"$ROOT/ocean/affine_lock" \
    "$SRC" -lm -o "$OUT"

"$CC_BIN" \
    -std=c11 -O0 -g \
    -Wall -Wextra -Werror \
    -I"$ROOT" -I"$ROOT/ocean/affine_lock" \
    "$LOADER_SRC" -lm -o "$LOADER_OUT"

"$OUT" \
    --sample-per-depth 128 \
    --output-bin "$BIN_OUT" \
    --output-json "$JSON_OUT"

"$LOADER_OUT" "$BIN_OUT" 101188 128 100548

"$OUT" \
    --sample-per-depth 8 \
    --sample-seed 42 \
    --output-bin "$SEED_42_A_BIN_OUT" \
    --output-json "$SEED_42_A_JSON_OUT"

"$OUT" \
    --sample-per-depth 8 \
    --sample-seed 42 \
    --output-bin "$SEED_42_B_BIN_OUT" \
    --output-json "$SEED_42_B_JSON_OUT"

"$OUT" \
    --sample-per-depth 8 \
    --sample-seed 69 \
    --output-bin "$SEED_69_BIN_OUT" \
    --output-json "$SEED_69_JSON_OUT"

cmp "$SEED_42_A_BIN_OUT" "$SEED_42_B_BIN_OUT"
if cmp -s "$SEED_42_A_BIN_OUT" "$SEED_69_BIN_OUT"; then
    echo "different sample seeds unexpectedly produced identical tables" >&2
    exit 1
fi

"$OUT" \
    --action-set affine_lock_4action_v1 \
    --sample-per-depth 16 \
    --output-bin "$FOUR_BIN_OUT" \
    --output-json "$FOUR_JSON_OUT"

python3 - \
    "$BIN_OUT" "$JSON_OUT" \
    "$SEED_42_A_BIN_OUT" "$SEED_42_A_JSON_OUT" \
    "$SEED_69_BIN_OUT" "$SEED_69_JSON_OUT" \
    "$FOUR_BIN_OUT" "$FOUR_JSON_OUT" <<'PY'
import json
import struct
import sys
from pathlib import Path

bin_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])
seed_42_bin_path = Path(sys.argv[3])
seed_42_json_path = Path(sys.argv[4])
seed_69_bin_path = Path(sys.argv[5])
seed_69_json_path = Path(sys.argv[6])
four_bin_path = Path(sys.argv[7])
four_json_path = Path(sys.argv[8])
manifest = json.loads(json_path.read_text())

assert manifest["action_set"] == "affine_lock_8action_v1"
assert manifest["action_id_to_name"] == [
    "shift_left",
    "shift_right",
    "invert_right_7",
    "swap_adjacent_bits",
    "swap_adjacent_pairs",
    "swap_nibbles_each_byte",
    "reverse_each_nibble",
    "reverse_each_byte",
]
assert manifest["bits"] == 16
assert manifest["num_actions"] == 8
assert manifest["depths"] == [2, 4, 5, 6, 8, 16]
assert manifest["sample_per_depth"] == 128
assert manifest["sample_seed"] == 0
assert manifest["stored_all_depths"] == [16]
assert manifest["max_distance"] == 20
assert manifest["disconnected_starts"] == 0
assert manifest["visible_distance_histogram"]["16"] == 100548
assert manifest["visible_distance_histogram"]["20"] == 4

depth_records = manifest["depth_records"]
assert [record["depth"] for record in depth_records] == [2, 4, 5, 6, 8, 16]
for record in depth_records[:5]:
    assert record["stored_count"] == 128
    assert record["exact_pair_count"] >= record["stored_count"]
assert depth_records[5]["stored_count"] == 100548
assert depth_records[5]["exact_pair_count"] == 100548

data = bin_path.read_bytes()
fixed_header = struct.Struct("<8sIIIIIIIQQ")
(
    magic,
    version,
    header_size,
    record_size,
    bits,
    num_actions,
    depth_count,
    record_count,
    checksum,
    action_set_hash,
) = fixed_header.unpack_from(data, 0)

assert magic == b"AL7TGT1\0"
assert version == 1
assert header_size == manifest["header_size"]
assert record_size == manifest["record_size"] == 16
assert bits == 16
assert num_actions == 8
assert depth_count == 6
assert record_count == sum(record["stored_count"] for record in depth_records)
assert checksum == int(manifest["checksum"], 16)
assert action_set_hash == int(manifest["action_set_hash"], 16)
assert len(data) == header_size + record_count * record_size

depth_struct = struct.Struct("<IIIIQ")
offset = fixed_header.size
for expected in depth_records:
    depth, first_record, stored_count, reserved, exact_pair_count = (
        depth_struct.unpack_from(data, offset)
    )
    offset += depth_struct.size
    assert reserved == 0
    assert depth == expected["depth"]
    assert first_record == expected["first_record"]
    assert stored_count == expected["stored_count"]
    assert exact_pair_count == expected["exact_pair_count"]

record_struct = struct.Struct("<HHQBBH")
records_start = header_size
first_start, first_target, first_packed, first_length, first_depth, reserved = (
    record_struct.unpack_from(data, records_start)
)
assert reserved == 0
assert first_start <= 0xffff
assert first_target <= 0xffff
assert first_length == first_depth
assert first_depth in {2, 4, 5, 6, 8, 16}
assert first_packed >= 0

seed_42_manifest = json.loads(seed_42_json_path.read_text())
seed_69_manifest = json.loads(seed_69_json_path.read_text())
assert seed_42_manifest["sample_seed"] == 42
assert seed_69_manifest["sample_seed"] == 69
assert seed_42_manifest["sample_per_depth"] == 8
assert seed_69_manifest["sample_per_depth"] == 8
assert seed_42_manifest["depth_records"] == seed_69_manifest["depth_records"]

seed_42_data = seed_42_bin_path.read_bytes()
seed_69_data = seed_69_bin_path.read_bytes()

def record_span(table_manifest, depth):
    record = next(
        record for record in table_manifest["depth_records"]
        if record["depth"] == depth
    )
    start = (
        table_manifest["header_size"] +
        record["first_record"] * table_manifest["record_size"]
    )
    end = start + record["stored_count"] * table_manifest["record_size"]
    return start, end

sampled_depths_changed = False
for depth in (2, 4, 5, 6, 8):
    start, end = record_span(seed_42_manifest, depth)
    if seed_42_data[start:end] != seed_69_data[start:end]:
        sampled_depths_changed = True
assert sampled_depths_changed

start, end = record_span(seed_42_manifest, 16)
assert seed_42_data[start:end] == seed_69_data[start:end]

four_manifest = json.loads(four_json_path.read_text())
assert four_manifest["action_set"] == "affine_lock_4action_v1"
assert four_manifest["action_id_to_name"] == [
    "shift_right",
    "mirror",
    "invert_right_7",
    "swap_adjacent_bits",
]
assert four_manifest["bits"] == 16
assert four_manifest["num_actions"] == 4
assert four_manifest["depths"] == [2, 4, 5, 6, 8, 16]
assert four_manifest["sample_per_depth"] == 16
assert four_manifest["sample_seed"] == 0
assert four_manifest["stored_all_depths"] == []
assert four_manifest["max_distance"] == 19
assert four_manifest["disconnected_starts"] == 0
assert four_manifest["visible_distance_histogram"]["16"] == 2434606
assert [record["stored_count"] for record in four_manifest["depth_records"]] == [
    16,
    16,
    16,
    16,
    16,
    16,
]
assert four_bin_path.stat().st_size == (
    four_manifest["header_size"] +
    four_manifest["record_count"] * four_manifest["record_size"]
)
PY
