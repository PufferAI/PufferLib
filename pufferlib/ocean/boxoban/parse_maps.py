import sys
import struct

# mapping
AGENT = '@'
WALL  = '#'
BOX   = '$'
TARG  = '.'
BOX_ON_TARG = '*'
AGENT_ON_TARG = '+'

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

def encode_puzzle(grid):
    # grid is 10 strings of length 10
    agent  = []
    walls  = []
    boxes  = []
    targ   = []

    for r in range(10):
        for c in range(10):
            ch = grid[r][c]
            agent.append(1 if ch in (AGENT, AGENT_ON_TARG) else 0)
            walls.append(1 if ch == WALL else 0)
            boxes.append(1 if ch in (BOX, BOX_ON_TARG) else 0)
            targ.append(1 if ch in (TARG, BOX_ON_TARG, AGENT_ON_TARG) else 0)

    return agent, walls, boxes, targ

def write_bin(files, out_path, verbose=True):
    all_arrays = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        puzzles = parse_puzzles(content)
        for p in puzzles:
            if len(p) != 10:
                raise ValueError(f"Puzzle not 10 lines in {path}")
            arrays = encode_puzzle(p)
            all_arrays.extend(arrays)

    # Flatten and write as bytes
    flat = bytearray()
    for arr in all_arrays:
        flat.extend(bytes(arr))

    count = len(all_arrays) // 4
    with open(out_path, "wb") as out:
        out.write(flat)

    if verbose:
        print(f"Wrote {count} puzzles to {out_path}")
    return count

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python build_bin.py out.bin file1.txt file2.txt ...")
        sys.exit(1)

    out = sys.argv[1]
    files = sys.argv[2:]
    write_bin(files, out)
