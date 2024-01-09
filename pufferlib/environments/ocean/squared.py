from pdb import set_trace as T

import numpy as np
import random
import gymnasium

from pufferlib import namespace


MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1)]

def all_possible_targets(n):
    '''Generate all points on the perimeter of a square with side n'''
    return ([(i, j) for i in range(n) for j in [0, n-1]]
        + [(i, j) for i in [0, n-1] for j in range(1, n-1)])

def init(self,
        distance_to_target=1,
        num_targets=-1,
        ):
    '''Pufferlib Diamond environment

    Agent starts at the center of a square grid.
    Targets are placed on the perimeter of the grid.
    Reward is 1 minus the L-inf distance to the closest target.
    This means that reward varies from -1 to 1.
    Reward is not given for targets that have already been hit.
    '''
    grid_size = 2 * distance_to_target + 1
    if num_targets == -1:
        num_targets = 4 * distance_to_target

    return namespace(self,
        distance_to_target=distance_to_target,
        possible_targets=all_possible_targets(grid_size),
        num_targets=num_targets,
        grid_size = grid_size,
        max_ticks = num_targets * distance_to_target,
        observation_space=gymnasium.spaces.Box(
            low=-1, high=1, shape=(grid_size, grid_size)),
        action_space=gymnasium.spaces.Discrete(8),
    )

def reset(state, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Allocating a new grid is faster than resetting an old one
    state.grid = np.zeros((state.grid_size, state.grid_size), dtype=np.float32)
    state.grid[state.distance_to_target, state.distance_to_target] = -1
    state.agent_pos = (state.distance_to_target, state.distance_to_target)
    state.tick = 0

    state.targets = random.sample(state.possible_targets, state.num_targets)
    for x, y in state.targets:
        state.grid[x, y] = 1

    return state.grid, {}

def step(state, action):
    x, y = state.agent_pos
    state.grid[x, y] = 0

    dx, dy = MOVES[action]
    x += dx
    y += dy

    min_dist = min([max(abs(x-tx), abs(y-ty)) for tx, ty in state.targets])
    # This reward function will return 0.46 average reward for an unsuccessful
    # episode with distance_to_target=4 and num_targets=1 (0.5 for solve)
    # It looks reasonable but is not very discriminative
    reward = 1 - min_dist / state.distance_to_target

    # This reward function will return 1 when the agent moves in the right direction
    # (plus an adjustment for the 0 reset reward) to average 1 for success
    # It is not much better than the previous one.
    #reward = state.distance_to_target - min_dist - state.tick + 1/state.max_ticks

    # This function will return 0, 0.2, 0.4, ... 1 for successful episodes (n=5)
    # And will drop rewards to 0 or less as soon as an error is made
    # Somewhat smoother but actually worse than the previous ones
    # reward = (state.distance_to_target - min_dist - state.tick) / (state.max_ticks - state.tick)


    # This one nicely tracks the task completed metric but does not optimize well
    #if state.distance_to_target - min_dist - state.tick  == 1:
    #    reward = 1
    #else:
    #    reward = -state.tick


    if (x, y) in state.targets:
        state.targets.remove((x, y))
        #state.grid[x, y] = 0

    dist_from_origin = max(abs(x-state.distance_to_target), abs(y-state.distance_to_target))
    if dist_from_origin >= state.distance_to_target:
        state.agent_pos = state.distance_to_target, state.distance_to_target
    else:
        state.agent_pos = x, y

    state.grid[state.agent_pos] = -1
    state.tick += 1

    done = state.tick >= state.max_ticks
    info = {'targets_hit': state.num_targets - len(state.targets)} if done else {}

    return state.grid, reward, done, False, info

def render(state):
    chars = []
    for row in state.grid:
        for val in row:
            if val == 1:
                color = 94 # Blue
            elif val == -1:
                color = 91 # Red
            else:
                color = 90 # Gray
            chars.append(f'\033[{color}m██\033[0m')
        chars.append('\n')
    return ''.join(chars)

def close(state):
    pass

class Squared(gymnasium.Env):
    __init__ = init
    reset = reset
    step = step
    render = render
    close = close
