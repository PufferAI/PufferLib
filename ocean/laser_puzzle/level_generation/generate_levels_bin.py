import os
import random
import struct
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from generate_level import generate_puzzle
from optimal_solver import iddfs


OUTPUT_PATH = Path(__file__).resolve().parents[3] / "resources" / "laser_puzzle" / "laser_puzzle_levels.bin"
MAGIC = b"LPZL"
VERSION = 1
WORKERS = max(1, (os.cpu_count() or 1) // 2)

GROUPS = {
    7: {"count": 8, "sensor_counts": {4}},
    6: {"count": 12, "sensor_counts": {4}},
    5: {"count": 25, "sensor_counts": {3, 4}},
    4: {"count": 25, "sensor_counts": {2, 3}},
    2: {"count": 25, "sensor_counts": {1, 2}},
    1: {"count": 5, "sensor_counts": {1}},
}

CELL_TYPE_EMPTY = 0
CELL_TYPE_LASER = 1
CELL_TYPE_SENSOR = 2
MIRROR_NONE = 0
MIRROR_RIGHT = 1
MIRROR_LEFT = 2

def strip_mirrors(grid):
    return [["*" if grid[r][c] in {"ML", "MR"} else grid[r][c] for c in range(len(grid[r]))] for r in range(len(grid))]

def sensor_count(grid):
    return sum(1 for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c].startswith("S"))

def encode_cell(token):
    if token == "*":
        return CELL_TYPE_EMPTY, MIRROR_NONE, -1
    if token == "MR":
        return CELL_TYPE_EMPTY, MIRROR_RIGHT, -1
    if token == "ML":
        return CELL_TYPE_EMPTY, MIRROR_LEFT, -1
    if token.startswith("L"):
        return CELL_TYPE_LASER, MIRROR_NONE, int(token[1:])
    if token.startswith("S"):
        return CELL_TYPE_SENSOR, MIRROR_NONE, int(token[1:])
    raise ValueError(f"Unknown cell token: {token}")


def write_levels_bin(levels, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        file.write(MAGIC)
        file.write(struct.pack("<II", VERSION, len(levels)))
        for level in levels:
            file.write(struct.pack("<ii", level["optimal_mirrors"], level["sensor_count"]))
            puzzle = level["puzzle"]
            for r in range(len(puzzle)):
                for c in range(len(puzzle[r])):
                    token = puzzle[r][c]
                    file.write(struct.pack("<BBbB", *encode_cell(token), 0))


def make_candidate(attempt, desired_optimal):
    sensor_counts = GROUPS[desired_optimal]["sensor_counts"]

    solved_grid, _tries = generate_puzzle(
        4, 4, 4, 4,
        min(sensor_counts), max(sensor_counts),
        desired_optimal, desired_optimal,
        100,
    )
    if solved_grid is None:
        return None

    puzzle = strip_mirrors(solved_grid)
    optimal_mirrors = iddfs(puzzle, 7)[1]
    sensors = sensor_count(puzzle)
    if optimal_mirrors not in GROUPS or sensors not in GROUPS[optimal_mirrors]["sensor_counts"]:
        return None

    return {
        "attempt": attempt,
        "puzzle": puzzle,
        "sensor_count": sensors,
        "optimal_mirrors": optimal_mirrors,
    }


def generate_verified_puzzles():
    found = {mirror_count: [] for mirror_count in GROUPS}
    seen_puzzles = set()

    def unfilled_groups():
        return [
            mirror_count for mirror_count in GROUPS
            if len(found[mirror_count]) < GROUPS[mirror_count]["count"]
        ]

    attempt = 0
    pending = set()

    def submit_candidate(executor):
        nonlocal attempt
        unfilled = unfilled_groups()
        attempt += 1
        desired_optimal = random.choice(unfilled)
        pending.add(executor.submit(make_candidate, attempt, desired_optimal))

    print(f"generating with {WORKERS} workers", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for _ in range(WORKERS):
            submit_candidate(executor)

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                candidate = future.result()
                if candidate is None:
                    continue

                puzzle = candidate["puzzle"]
                optimal_mirrors = candidate["optimal_mirrors"]
                sensors = candidate["sensor_count"]

                puzzle_key = tuple(tuple(row) for row in puzzle)
                if puzzle_key in seen_puzzles or len(found[optimal_mirrors]) >= GROUPS[optimal_mirrors]["count"]:
                    continue

                seen_puzzles.add(puzzle_key)
                found[optimal_mirrors].append(candidate)
                print(
                    "found",
                    optimal_mirrors,
                    f"{len(found[optimal_mirrors])}/{GROUPS[optimal_mirrors]['count']}",
                    "sensors",
                    sensors,
                    "attempts",
                    candidate["attempt"],
                    flush=True,
                )

            if not unfilled_groups():
                for future in pending:
                    future.cancel()
                break
            while len(pending) < WORKERS and unfilled_groups():
                submit_candidate(executor)

    return [level for mirror_count in GROUPS for level in found[mirror_count]]


def main():
    levels = generate_verified_puzzles()
    write_levels_bin(levels, OUTPUT_PATH)
    print(f"wrote {len(levels)} levels to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
