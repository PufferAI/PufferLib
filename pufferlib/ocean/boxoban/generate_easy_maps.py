import random
from pathlib import Path

AGENT = '@'
WALL = '#'
BOX = '$'
TARGET = '.'
FLOOR = ' '
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def build_border_grid(size):
    grid = [[FLOOR for _ in range(size)] for _ in range(size)]
    for idx in range(size):
        grid[0][idx] = WALL
        grid[size - 1][idx] = WALL
        grid[idx][0] = WALL
        grid[idx][size - 1] = WALL
    return grid


def interior_cells(size, margin = 0):
    return [(r, c) 
            for r in range(1 + margin, size - 1 - margin) 
            for c in range(1 + margin, size - 1 - margin)
            ]


def is_inside(size, x, y):
    return 0 <= x < size and 0 <= y < size


def is_pushable(grid, size, x, y):
    for dx, dy in DIRECTIONS:
        px, py = x - dx, y - dy
        tx, ty = x + dx, y + dy
        if not (is_inside(size, px, py) and is_inside(size, tx, ty)):
            continue
        if grid[py][px] in (FLOOR, TARGET) and grid[ty][tx] in (FLOOR, TARGET):
            return True
    return False


def make_puzzle(size, rng, num_boxes, max_attempts=200):
    if num_boxes < 1:
        raise ValueError("num_boxes must be at least 1")

    agent_choices = interior_cells(size)
    interior = interior_cells(size)
    confined = interior_cells(size, margin=1)
    needed = num_boxes * 2 + 1  # targets + boxes + agent
    if needed > len(confined) + (len(agent_choices) - len(confined)):
        raise ValueError(
            f"Grid interior only has {len(interior)} cells, cannot place {needed} objects"
        )

    for _ in range(max_attempts):
        agent_pos = rng.choice(agent_choices)
        confined_samples = rng.sample(confined, num_boxes * 2)
        target_positions = confined_samples[:num_boxes]
        box_positions = confined_samples[num_boxes:]

       
        grid = build_border_grid(size)
        for tr, tc in target_positions:
            grid[tr][tc] = TARGET
        for br, bc in box_positions:
            grid[br][bc] = BOX

        if all(is_pushable(grid, size, bc, br) for br, bc in box_positions):
            ar, ac = agent_pos
            grid[ar][ac] = AGENT
            return [''.join(row) for row in grid]

    raise RuntimeError("Failed to sample a solvable puzzle after many attempts")


def write_text_file(puzzles, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        for idx, puzzle in enumerate(puzzles):
            handle.write(f"; {idx}\n")
            for line in puzzle:
                handle.write(line + '\n')
            handle.write('\n')


def main():
    num_files = 300
    puzzles_per_file = 1000
    size = 10
    num_boxes = None  # set to an int to fix boxes/targets per map
    min_boxes = 1
    max_boxes = 1
    seed = 0
    output_dir = Path("boxoban-levels/basic/train")

    rng = random.Random(seed)
    for file_idx in range(num_files):
        puzzles = []
        for _ in range(puzzles_per_file):
            if num_boxes is not None:
                box_count = num_boxes
            else:
                box_count = rng.randint(min_boxes, max_boxes)
            puzzles.append(make_puzzle(size, rng, box_count))

        output_path = output_dir / f"{file_idx:03d}.txt"
        write_text_file(puzzles, output_path)


if __name__ == "__main__":
    main()
