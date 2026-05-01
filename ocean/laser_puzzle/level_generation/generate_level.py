import random

DIRS_MAP = {
    "down": (1, 0),
    "up": (-1, 0),
    "right": (0, 1),
    "left": (0, -1),
}
MIRRORS = {"MR", "ML"}

# returns a grid with one higher in each dimension (to store the border for the lasers sources and sinks)
def generate_grid(
    MIN_ROWS,
    MAX_ROWS,
    MIN_COLS,
    MAX_COLS,
    MIN_LASERS,
    MAX_LASERS,
):
    puzzle_rows = random.randint(MIN_ROWS, MAX_ROWS)
    puzzle_cols = random.randint(MIN_COLS, MAX_COLS)
    possible_lasers = random.randint(MIN_LASERS, MAX_LASERS)

    # we will augment the grid by one to store the laser sources and sinks on the border
    grid = [['*'] * (puzzle_cols + 2) for _ in range(puzzle_rows + 2)]

    # choose where to put the lasers
    ROWS, COLS = len(grid), len(grid[0])
    laser_choices = (
        [(0, c) for c in range(1, COLS - 1)] +
        [(ROWS - 1, c) for c in range(1, COLS - 1)] +
        [(r, 0) for r in range(1, ROWS - 1)] +
        [(r, COLS - 1) for r in range(1, ROWS - 1)]
    )

    # place lasers
    laser_count = min(len(laser_choices), possible_lasers)
    for idx, pos in enumerate(random.sample(laser_choices, laser_count)):
        grid[pos[0]][pos[1]] = f"L{idx}"

    return grid

def on_border(pos, grid):
    return pos[0] in (0, len(grid) - 1) or pos[1] in (0, len(grid[0]) - 1)

def laser_positions(grid):
    return {
        (r, c)
        for r, row in enumerate(grid)
        for c, cell in enumerate(row)
        if cell.startswith("L")
    }

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
def laser_step(
    pos,
    direction,
    grid,
):
    # take a step and check if we are at end / hit a mirror
    nr = pos[0] + direction[0]
    nc = pos[1] + direction[1]

    # we have hit the border
    if on_border((nr, nc), grid):
        return (nr, nc), None

    # check if we hit a mirror and reflect
    if grid[nr][nc] == "ML":
        return (nr, nc), (direction[1], direction[0])

    if grid[nr][nc] == "MR":
        return (nr, nc), (-direction[1], -direction[0])

    # no mirror
    return (nr, nc), direction

def walk_laser(grid, start):
    pos = start
    direction = laser_direction(start, grid)
    while direction is not None:
        pos, direction = laser_step(pos, direction, grid)
        yield pos, direction

def laser_sink(grid, start):
    for pos, direction in walk_laser(grid, start):
        if direction is None:
            return pos

# returns if the current grid follows all the rules
def ensure_rules(grid):
    mirrors = {
        (r, c)
        for r, row in enumerate(grid)
        for c, cell in enumerate(row)
        if cell in MIRRORS
    }
    visited_lasers = set()

    for start in laser_positions(grid):
        start_state = (start, laser_direction(start, grid))
        visited_lasers.add(start_state)

        for pos, direction in walk_laser(grid, start):
            if direction is None:
                break

            state = (pos, direction)
            if state in visited_lasers:
                return False

            mirrors.discard(pos)
            visited_lasers.add(state)

    # all mirrors must be used
    return not mirrors


#  expects a valid grid that follows the rules in ensure_rules
def place_a_mirror(grid):
    valid = []
    for start in laser_positions(grid):
        for pos, direction in walk_laser(grid, start):
            if direction is None:
                break
            if grid[pos[0]][pos[1]] == '*':
                valid.append(pos)

    # Try valid slots in random order.
    # Duplicates bias toward cells hit by multiple lasers since using a list instead of a set
    random.shuffle(valid)

    for chosen in valid:
        orientations = ["ML", "MR"]
        random.shuffle(orientations)

        for orientation in orientations:
            grid[chosen[0]][chosen[1]] = orientation
            if ensure_rules(grid):
                return True

            # backtrack
            grid[chosen[0]][chosen[1]] = '*'

    return False


def generate_puzzle(
    MIN_ROWS,
    MAX_ROWS,
    MIN_COLS,
    MAX_COLS,
    MIN_LASERS,
    MAX_LASERS,
    MIN_MIRRORS,
    MAX_MIRRORS,
    MAX_TRIES,
):

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

    lasers = laser_positions(grid)
    sinks = set()
    insert_sinks = []
    for start in lasers:
        sink = laser_sink(grid, start)

        if sink in sinks or sink in lasers:
            return None, None

        laser_number = grid[start[0]][start[1]][1:]
        sinks.add(sink)
        insert_sinks.append((sink, laser_number))

    # now make sure the source, sink pairs are paired and labelled correctly in the graph
    for sink, laser_number in insert_sinks:
        grid[sink[0]][sink[1]] = f"S{laser_number}"

    return grid, tries + 1