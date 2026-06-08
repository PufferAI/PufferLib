import random

DIRS_MAP = {
    "down": (1, 0),
    "up": (-1, 0),
    "right": (0, 1),
    "left": (0, -1),
}
MIRRORS = {"MR", "ML"}

# returns a grid with one higher in each dimension (to store the border for the lasers sources and sinks)
def generate_grid(MIN_ROWS, MAX_ROWS, MIN_COLS, MAX_COLS, MIN_LASERS, MAX_LASERS):
    puzzle_rows = random.randint(MIN_ROWS, MAX_ROWS)
    puzzle_cols = random.randint(MIN_COLS, MAX_COLS)
    possible_lasers = random.randint(MIN_LASERS, MAX_LASERS)

    # we will augment the grid by one to store the laser sources and sinks on the border
    grid = [['*'] * (puzzle_cols + 2) for _ in range(puzzle_rows + 2)]

    # choose where to put the lasers (inner cells only)
    ROWS, COLS = len(grid), len(grid[0])
    laser_choices = (
        [(0, c) for c in range(1, COLS - 1)] +
        [(ROWS - 1, c) for c in range(1, COLS - 1)] +
        [(r, 0) for r in range(1, ROWS - 1)] +
        [(r, COLS - 1) for r in range(1, ROWS - 1)]
    )

    laser_count = min(len(laser_choices), possible_lasers)
    for idx, pos in enumerate(random.sample(laser_choices, laser_count)):
        grid[pos[0]][pos[1]] = f"L{idx}"

    return grid

def on_border(pos, grid):
    return pos[0] in (0, len(grid) - 1) or pos[1] in (0, len(grid[0]) - 1)

def laser_direction(pos, grid):
    rows = len(grid)
    if pos[0] == 0:
        return DIRS_MAP["down"]
    if pos[0] == rows - 1:
        return DIRS_MAP["up"]
    if pos[1] == 0:
        return DIRS_MAP["right"]
    return DIRS_MAP["left"]

# take a step with laser, give new pos of laser, give new direction of laser also (None if at border)
def laser_step(pos, direction, grid):
    nr = pos[0] + direction[0]
    nc = pos[1] + direction[1]

    # we have hit the border
    if on_border((nr, nc), grid):
        return (nr, nc), None

    # check if we hit a mirror and reflect
    if grid[nr][nc] == "ML":
        return (nr, nc), (direction[1], direction[0])
    elif grid[nr][nc] == "MR":
        return (nr, nc), (-direction[1], -direction[0])

    # no mirror, no border, just empty cell
    return (nr, nc), direction

# give one by one, the next position of a laser, taking into acount the board state
def walk_laser(grid, start):
    pos = start
    direction = laser_direction(start, grid)
    while direction is not None:
        pos, direction = laser_step(pos, direction, grid)
        yield pos, direction

# Rules:
# 1) laser paths cannot cycle
# 2) laser paths of two colors can never be in the same position and same direction
# 3) every mirror must be hit by at least one laser
def ensure_rules(grid):
    mirrors = {(r, c) for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c] in MIRRORS}

    visited_lasers = set()

    for laser_start in {(r, c) for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c].startswith("L")}:
        start_state = (laser_start, laser_direction(laser_start, grid))
        visited_lasers.add(start_state)

        for pos, direction in walk_laser(grid, laser_start):
            if direction is None:
                break

            state = (pos, direction)
            if state in visited_lasers:
                return False

            mirrors.discard(pos)
            visited_lasers.add(state)

    # all mirrors must be used
    return not mirrors


def place_a_mirror(grid):
    valid = []
    for laser_start in {(r, c) for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c].startswith("L")}:
        for pos, direction in walk_laser(grid, laser_start):
            if direction is None:
                break
            if grid[pos[0]][pos[1]] == '*':
                valid.append(pos)

    # Try valid slots in random order. Duplicates bias toward cells hit by multiple lasers since using a list instead of a set, good for puzzle complexity
    random.shuffle(valid)

    for chosen in valid:
        orientations = ["ML", "MR"]
        random.shuffle(orientations)

        for orientation in orientations:
            grid[chosen[0]][chosen[1]] = orientation
            if ensure_rules(grid):
                return True
            else:
                # backtrack
                grid[chosen[0]][chosen[1]] = '*'

    return False


def generate_puzzle(MIN_ROWS, MAX_ROWS, MIN_COLS, MAX_COLS, MIN_LASERS, MAX_LASERS, MIN_MIRRORS, MAX_MIRRORS, MAX_TRIES):

    need_mirrors = random.randint(MIN_MIRRORS, MAX_MIRRORS)
    for tries in range(MAX_TRIES):
        # create a fresh grid for this full attempt
        grid = generate_grid(MIN_ROWS, MAX_ROWS, MIN_COLS, MAX_COLS, MIN_LASERS, MAX_LASERS)

        for _ in range(need_mirrors):
            if not place_a_mirror(grid):
                break
        else:
            break
    else:
        return None, None

    lasers = {(r, c) for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c].startswith("L")}
    sinks = set()
    insert_sinks = []
    for laser_start in lasers:
        laser_pos = None
        for laser_pos, direction in walk_laser(grid, laser_start):
            if direction is None:
                break

        if laser_pos in sinks or laser_pos in lasers:
            return None, None

        laser_number = grid[laser_start[0]][laser_start[1]][1:]
        sinks.add(laser_pos)
        insert_sinks.append((laser_pos, laser_number))

    # now make sure the source, sink pairs are paired and labelled correctly in the graph
    for sink, laser_number in insert_sinks:
        grid[sink[0]][sink[1]] = f"S{laser_number}"

    return grid, tries + 1
