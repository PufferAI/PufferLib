import sys
import struct

# mapping
AGENT = '@'
WALL  = '#'
BOX   = '$'
TARG  = '.'
BOX_ON_TARG = '*'
AGENT_ON_TARG = '+'
PUZZLE_OBS_BYTES = 4 * 10 * 10
PUZZLE_META_BYTES = 5
PUZZLE_BYTES = PUZZLE_OBS_BYTES + PUZZLE_META_BYTES
EXPECTED_ROWS = 10
EXPECTED_COLS = 10

def parse_puzzles(text):
    puzzles = []
    current = []

    for line in text.splitlines():
        line = line.rstrip('\n')
        if line.startswith(';'):
            if current:
                puzzles.append(current)
                current = []
            continue
        if line.strip() == "":
            continue
        current.append(line)

        if len(current) == 10:
            puzzles.append(current)
            current = []

    return puzzles

def validate_puzzle_shape(grid):
    if len(grid) != EXPECTED_ROWS:
        return False, f"expected {EXPECTED_ROWS} rows, got {len(grid)}"

    for r, row in enumerate(grid):
        if len(row) != EXPECTED_COLS:
            return False, f"row {r} expected {EXPECTED_COLS} cols, got {len(row)}"

    return True, ""

def encode_puzzle(grid):
    # grid is 10 strings of length 10
    agent  = []
    walls  = []
    boxes  = []
    targ   = []
    agent_x = -1
    agent_y = -1
    n_boxes = 0
    n_targets = 0
    on_target = 0

    for r in range(10):
        for c in range(10):
            ch = grid[r][c]
            is_agent = ch in (AGENT, AGENT_ON_TARG)
            is_wall = ch == WALL
            is_box = ch in (BOX, BOX_ON_TARG)
            is_target = ch in (TARG, BOX_ON_TARG, AGENT_ON_TARG)

            if is_agent:
                if agent_x != -1:
                    raise ValueError("Puzzle has multiple agents")
                agent_x = c
                agent_y = r

            n_boxes += int(is_box)
            n_targets += int(is_target)
            on_target += int(is_box and is_target)

            agent.append(1 if is_agent else 0)
            walls.append(1 if is_wall else 0)
            boxes.append(1 if is_box else 0)
            targ.append(1 if is_target else 0)

    if agent_x == -1:
        raise ValueError("Puzzle has no agent")

    meta = [agent_x, agent_y, n_boxes, n_targets, on_target]
    return agent, walls, boxes, targ, meta

def write_bin(files, out_path, verbose=True):
    all_arrays = []
    puzzle_count = 0

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        puzzles = parse_puzzles(content)
        for idx, p in enumerate(puzzles):
            ok, reason = validate_puzzle_shape(p)
            if not ok:
                print(f"[Boxoban] Skipping malformed puzzle in {path} puzzle#{idx}: {reason}")
                continue

            try:
                arrays = encode_puzzle(p)
            except ValueError as e:
                print(f"[Boxoban] Skipping malformed puzzle in {path} puzzle#{idx}: {e}")
                continue

            all_arrays.extend(arrays)
            puzzle_count += 1

    # Flatten and write as bytes
    flat = bytearray()
    for arr in all_arrays:
        flat.extend(bytes(arr))

    with open(out_path, "wb") as out:
        out.write(flat)

    expected_size = puzzle_count * PUZZLE_BYTES
    if len(flat) != expected_size:
        raise ValueError(f"Wrong output size: got {len(flat)} expected {expected_size}")

    if verbose:
        print(f"Wrote {puzzle_count} puzzles to {out_path}")
    return puzzle_count

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python build_bin.py out.bin file1.txt file2.txt ...")
        sys.exit(1)

    out = sys.argv[1]
    files = sys.argv[2:]
    write_bin(files, out)
