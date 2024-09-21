import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t
import random

cnp.import_array()  # Initialize NumPy's C-API

# Core environment structure to manage grid-based logic
cdef struct CSquaredEnv:
    int grid_size
    int distance_to_target
    int num_targets
    int max_ticks
    int tick
    int* agent_pos  # For agent's x, y position
    float* grid  # Grid, stored as a flat array (1D C array)
    int* targets  # C array to store target positions as pairs of ints (x, y)

# Initialize the environment with the given parameters
cdef CSquaredEnv* init_squared_env(int distance_to_target, int num_targets):
    cdef int grid_size = 2 * distance_to_target + 1
    cdef CSquaredEnv* env = <CSquaredEnv*> malloc(sizeof(CSquaredEnv))
    env.grid_size = grid_size
    env.distance_to_target = distance_to_target
    env.num_targets = num_targets
    env.max_ticks = num_targets * distance_to_target
    env.tick = 0
    env.agent_pos = <int*> malloc(2 * sizeof(int))  # Store (x, y) as 2 integers
    env.grid = <float*> malloc(grid_size * grid_size * sizeof(float))  # Grid
    env.targets = <int*> malloc(num_targets * 2 * sizeof(int))  # Store targets as pairs of (x, y)
    return env

cdef void reset(CSquaredEnv* env):
    env.tick = 0
    cdef int grid_size = env.grid_size
    cdef int distance_to_target = env.distance_to_target

    # Clear grid
    for i in range(grid_size * grid_size):
        env.grid[i] = 0.0

    # Place agent in the center
    env.agent_pos[0] = distance_to_target
    env.agent_pos[1] = distance_to_target
    env.grid[distance_to_target * grid_size + distance_to_target] = -1.0

    # Set random targets along the grid perimeter
    possible_targets = [(x, y) for x in range(grid_size) for y in range(grid_size)
                        if x == 0 or y == 0 or x == grid_size - 1 or y == grid_size - 1]
    selected_targets = random.sample(possible_targets, env.num_targets)
    
    for i, (tx, ty) in enumerate(selected_targets):
        env.targets[i * 2] = tx  # Store target x
        env.targets[i * 2 + 1] = ty  # Store target y
        env.grid[tx * grid_size + ty] = 1.0  # Mark target on grid

cdef tuple step(CSquaredEnv* env, int action):
    cdef int x = env.agent_pos[0]
    cdef int y = env.agent_pos[1]
    cdef int grid_size = env.grid_size
    cdef int distance_to_target = env.distance_to_target
    cdef int dx, dy
    cdef list moves = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1)]

    # Clear previous agent position on grid
    env.grid[x * grid_size + y] = 0.0

    # Move agent
    dx, dy = moves[action]
    x += dx
    y += dy

    # Calculate the minimum distance to any target
    cdef int i, tx, ty
    cdef float min_dist = 1e9  # Large initial value
    for i in range(env.num_targets):
        tx = env.targets[i * 2]
        ty = env.targets[i * 2 + 1]
        if tx >= 0 and ty >= 0:  # Only check valid targets
            min_dist = min(min_dist, max(abs(x - tx), abs(y - ty)))

    reward = 1.0 - (min_dist / distance_to_target)

    # Check if agent hits a target
    for i in range(env.num_targets):
        tx = env.targets[i * 2]
        ty = env.targets[i * 2 + 1]
        if x == tx and y == ty:
            env.targets[i * 2] = -1  # Mark target as hit
            env.targets[i * 2 + 1] = -1

    # Ensure agent doesn't move out of bounds
    dist_from_origin = max(abs(x - distance_to_target), abs(y - distance_to_target))
    if dist_from_origin >= distance_to_target:
        env.agent_pos[0], env.agent_pos[1] = distance_to_target, distance_to_target
    else:
        env.agent_pos[0], env.agent_pos[1] = x, y

    # Mark agent's new position
    env.grid[env.agent_pos[0] * grid_size + env.agent_pos[1]] = -1.0
    env.tick += 1

    # Check if episode is done
    done = env.tick >= env.max_ticks

    # Calculate score (based on how many targets have been hit)
    hit_targets = 0
    for i in range(env.num_targets):
        if env.targets[i * 2] == -1:
            hit_targets += 1

    score = hit_targets / env.num_targets
    info = {'score': score} if done else {}

    return reward, done, info

# Free the memory for the environment
cdef void free_squared_env(CSquaredEnv* env):
    free(<void*> env.agent_pos)
    free(<void*> env.grid)
    free(<void*> env.targets)
    free(env)

cdef class CSquaredCy:
    cdef:
        CSquaredEnv* env
        int grid_size
        int distance_to_target
        int num_targets

    def __init__(self, int distance_to_target, int num_targets):
        self.grid_size = 2 * distance_to_target + 1
        self.distance_to_target = distance_to_target
        self.num_targets = num_targets

        # Initialize the environment
        self.env = init_squared_env(distance_to_target, num_targets)

    def reset(self):
        reset(self.env)

    def step(self, int action):
        return step(self.env, action)

    def get_grid(self):
        # Convert the C array to a memoryview and then to a NumPy array
        cdef int grid_size = self.grid_size
        cdef float[:] grid_view = <float[:grid_size * grid_size]> self.env.grid
        return np.asarray(grid_view).reshape(grid_size, grid_size)

    def __dealloc__(self):
        free_squared_env(self.env)