#!/usr/bin/env python3
"""Generate the affine-lock precomputed transform table."""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path


BITS = 16
NUM_ACTIONS = 8
MASK = (1 << BITS) - 1
EXPECTED_TRANSFORM_COUNT = 16384
EXPECTED_HISTOGRAM = {
    0: 1,
    1: 8,
    2: 25,
    3: 54,
    4: 104,
    5: 192,
    6: 346,
    7: 610,
    8: 1057,
    9: 1774,
    10: 2775,
    11: 3876,
    12: 3192,
    13: 1438,
    14: 646,
    15: 234,
    16: 46,
    17: 6,
}

AFFINE_DIR = Path(__file__).resolve().parents[1]
GENERATED_DIR = AFFINE_DIR / "generated"
HEADER_PATH = GENERATED_DIR / "affine_lock_transform_table.h"
MANIFEST_PATH = GENERATED_DIR / "affine_lock_transform_table.json"


def position_mask(start: int, stride: int) -> int:
    mask = 0
    for bit in range(start, BITS, stride):
        mask |= 1 << bit
    return mask


def range_mask(start: int, end: int) -> int:
    mask = 0
    for bit in range(start, end):
        mask |= 1 << bit
    return mask


def shift_left_perm() -> tuple[int, ...]:
    return tuple((out_bit + 1) % BITS for out_bit in range(BITS))


def shift_right_perm() -> tuple[int, ...]:
    return tuple((out_bit - 1) % BITS for out_bit in range(BITS))


def mirror_perm() -> tuple[int, ...]:
    return tuple(BITS - 1 - out_bit for out_bit in range(BITS))


def action_transform(action: int) -> tuple[tuple[int, ...], int]:
    identity = tuple(range(BITS))
    if action == 0:
        return identity, MASK
    if action == 1:
        return shift_left_perm(), 0
    if action == 2:
        return shift_right_perm(), 0
    if action == 3:
        return mirror_perm(), 0
    if action == 4:
        return identity, position_mask(0, 2)
    if action == 5:
        return identity, position_mask(1, 2)
    if action == 6:
        return identity, range_mask(0, BITS // 2)
    if action == 7:
        return identity, range_mask(BITS // 2, BITS)
    raise ValueError(f"invalid action {action}")


ACTION_TRANSFORMS = [action_transform(action) for action in range(NUM_ACTIONS)]


def permute_bits(value: int, perm: tuple[int, ...]) -> int:
    out = 0
    for out_bit, in_bit in enumerate(perm):
        if value & (1 << in_bit):
            out |= 1 << out_bit
    return out & MASK


def compose_after_action(
    perm: tuple[int, ...],
    xor_mask: int,
    action: int,
) -> tuple[tuple[int, ...], int]:
    action_perm, action_mask = ACTION_TRANSFORMS[action]
    next_perm = tuple(perm[action_perm[out_bit]] for out_bit in range(BITS))
    next_mask = permute_bits(xor_mask, action_perm) ^ action_mask
    return next_perm, next_mask & MASK


def apply_transform(state: int, perm: tuple[int, ...], xor_mask: int) -> int:
    return permute_bits(state, perm) ^ xor_mask


def apply_action(state: int, action: int) -> int:
    perm, xor_mask = ACTION_TRANSFORMS[action]
    return apply_transform(state, perm, xor_mask)


def pack_actions(actions: tuple[int, ...]) -> int:
    packed = 0
    for index, action in enumerate(actions):
        packed |= action << (3 * index)
    return packed


def unpack_actions(packed: int, count: int) -> tuple[int, ...]:
    return tuple((packed >> (3 * index)) & 7 for index in range(count))


def build_records() -> tuple[list[dict[str, int]], list[tuple[int, ...]], list[int]]:
    identity_perm = tuple(range(BITS))
    identity = (identity_perm, 0)
    queue = deque([identity])
    seen = {identity: tuple()}
    records: list[dict[str, int]] = []

    while queue:
        perm, xor_mask = queue.popleft()
        path = seen[(perm, xor_mask)]
        records.append(
            {
                "perm_tuple_index": -1,
                "xor_mask": xor_mask,
                "distance": len(path),
                "action_count": len(path),
                "packed_actions": pack_actions(path),
            }
        )

        for action in range(NUM_ACTIONS):
            next_key = compose_after_action(perm, xor_mask, action)
            if next_key in seen:
                continue
            seen[next_key] = path + (action,)
            queue.append(next_key)

    perms = sorted({perm for perm, _ in seen})
    perm_ids = {perm: index for index, perm in enumerate(perms)}
    shell_offsets = [0]
    histogram = Counter(record["distance"] for record in records)
    max_distance = max(histogram)
    for distance in range(max_distance + 1):
        shell_offsets.append(shell_offsets[-1] + histogram[distance])

    for record, (perm, _xor_mask) in zip(records, seen.keys(), strict=True):
        record["perm_tuple_index"] = perm_ids[perm]

    validate_records(records, perms, shell_offsets)
    return records, perms, shell_offsets


def validate_records(
    records: list[dict[str, int]],
    perms: list[tuple[int, ...]],
    shell_offsets: list[int],
) -> None:
    if len(records) != EXPECTED_TRANSFORM_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_TRANSFORM_COUNT} transforms, got {len(records)}"
        )
    histogram = Counter(record["distance"] for record in records)
    if dict(histogram) != EXPECTED_HISTOGRAM:
        raise RuntimeError(f"unexpected histogram: {dict(histogram)}")
    if len(perms) != 32:
        raise RuntimeError(f"expected 32 permutations, got {len(perms)}")
    if shell_offsets[-1] != len(records):
        raise RuntimeError("shell offsets do not cover all records")

    previous_distance = 0
    sample_states = (0x0000, 0x0001, 0xA55A, 0xF00F)
    for record in records:
        distance = record["distance"]
        if distance < previous_distance:
            raise RuntimeError("records are not sorted by distance")
        previous_distance = distance
        if record["action_count"] != distance:
            raise RuntimeError("record action count differs from distance")
        actions = unpack_actions(record["packed_actions"], record["action_count"])
        if any(action < 0 or action >= NUM_ACTIONS for action in actions):
            raise RuntimeError("packed action outside action range")
        perm = perms[record["perm_tuple_index"]]
        for state in sample_states:
            replayed = state
            for action in actions:
                replayed = apply_action(replayed, action)
            transformed = apply_transform(state, perm, record["xor_mask"])
            if replayed != transformed:
                raise RuntimeError("packed path does not replay transform")


def checksum_records(
    records: list[dict[str, int]],
    perms: list[tuple[int, ...]],
    shell_offsets: list[int],
) -> int:
    checksum = 1469598103934665603

    def mix(value: int) -> None:
        nonlocal checksum
        checksum ^= value & 0xFFFFFFFFFFFFFFFF
        checksum = (checksum * 1099511628211) & 0xFFFFFFFFFFFFFFFF

    for perm in perms:
        for bit in perm:
            mix(bit)
    for offset in shell_offsets:
        mix(offset)
    for record in records:
        mix(record["packed_actions"])
        mix(record["xor_mask"])
        mix(record["perm_tuple_index"])
        mix(record["distance"])
        mix(record["action_count"])
    return checksum


def render_header(
    records: list[dict[str, int]],
    perms: list[tuple[int, ...]],
    shell_offsets: list[int],
    checksum: int,
) -> str:
    lines: list[str] = [
        "#pragma once",
        "",
        "/* Generated by ocean/affine_lock/tools/generate_transform_table.py. */",
        "",
        "#include <stdint.h>",
        "",
        "#define AFFINE_LOCK_PRECOMPUTED_TRANSFORM_COUNT 16384",
        "#define AFFINE_LOCK_PRECOMPUTED_TRANSFORM_PERM_COUNT 32",
        "#define AFFINE_LOCK_PRECOMPUTED_TRANSFORM_MAX_DISTANCE 17",
        "#define AFFINE_LOCK_PRECOMPUTED_TRANSFORM_SHELL_OFFSET_COUNT 19",
        f"#define AFFINE_LOCK_PRECOMPUTED_TRANSFORM_CHECKSUM 0x{checksum:016x}ull",
        "",
        "typedef struct AffineLockPrecomputedTransform {",
        "    uint64_t packed_actions;",
        "    uint16_t xor_mask;",
        "    uint8_t perm_id;",
        "    uint8_t distance;",
        "    uint8_t action_count;",
        "} AffineLockPrecomputedTransform;",
        "",
        "static const uint8_t",
        "AFFINE_LOCK_PRECOMPUTED_TRANSFORM_PERMS[32][16] = {",
    ]

    for perm in perms:
        values = ", ".join(str(bit) for bit in perm)
        lines.append(f"    {{{values}}},")
    lines.extend(["};", ""])

    offset_values = ", ".join(str(offset) for offset in shell_offsets)
    lines.extend(
        [
            "static const uint32_t",
            "AFFINE_LOCK_PRECOMPUTED_TRANSFORM_SHELL_OFFSETS[19] = {",
            f"    {offset_values},",
            "};",
            "",
            "static const AffineLockPrecomputedTransform",
            "AFFINE_LOCK_PRECOMPUTED_TRANSFORMS[16384] = {",
        ]
    )
    for record in records:
        lines.append(
            "    "
            f"{{0x{record['packed_actions']:016x}ull, "
            f"0x{record['xor_mask']:04x}u, "
            f"{record['perm_tuple_index']}, "
            f"{record['distance']}, "
            f"{record['action_count']}}},"
        )
    lines.extend(["};", ""])
    return "\n".join(lines)


def render_manifest(
    records: list[dict[str, int]],
    perms: list[tuple[int, ...]],
    shell_offsets: list[int],
    checksum: int,
) -> str:
    histogram = Counter(record["distance"] for record in records)
    manifest = {
        "bits": BITS,
        "num_actions": NUM_ACTIONS,
        "transform_count": len(records),
        "perm_count": len(perms),
        "max_distance": max(histogram),
        "distance_histogram": {
            str(distance): histogram[distance] for distance in sorted(histogram)
        },
        "shell_offsets": shell_offsets,
        "checksum": f"0x{checksum:016x}",
    }
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def write_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text() == text:
        return
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated files are present and up to date",
    )
    args = parser.parse_args()

    records, perms, shell_offsets = build_records()
    checksum = checksum_records(records, perms, shell_offsets)
    header = render_header(records, perms, shell_offsets, checksum)
    manifest = render_manifest(records, perms, shell_offsets, checksum)

    if args.check:
        missing = [
            str(path)
            for path in (HEADER_PATH, MANIFEST_PATH)
            if not path.exists()
        ]
        if missing:
            print("missing generated affine lock transform artifacts:")
            for path in missing:
                print(f"  {path}")
            print("run: python3 ocean/affine_lock/tools/generate_transform_table.py")
            return 1
        failures = []
        if HEADER_PATH.read_text() != header:
            failures.append(str(HEADER_PATH))
        if MANIFEST_PATH.read_text() != manifest:
            failures.append(str(MANIFEST_PATH))
        if failures:
            print("stale generated affine lock transform artifacts:")
            for path in failures:
                print(f"  {path}")
            print("run: python3 ocean/affine_lock/tools/generate_transform_table.py")
            return 1
        return 0

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    write_if_changed(HEADER_PATH, header)
    write_if_changed(MANIFEST_PATH, manifest)
    print(f"wrote {HEADER_PATH}")
    print(f"wrote {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
