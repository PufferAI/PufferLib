import gymnasium
import numpy as np
import random

from pufferlib import namespace


def init(self,
        mem_length=1,
        mem_delay=0,
        ):
    '''Pufferlib Memory environment

    Repeat the provided sequence back
    '''
    return namespace(self,
        mem_length=mem_length,
        mem_delay=mem_delay,
        horizon = 2 * mem_length + mem_delay,
        observation_space=gymnasium.spaces.Box(
            low=-1, high=1, shape=(1,)),
        action_space=gymnasium.spaces.Discrete(2),
    )

def reset(state, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    state.solution = np.random.randint(0, 2, size=state.horizon).astype(np.float32)
    state.solution[-(state.mem_length + state.mem_delay):] = -1
    state.submission = np.zeros(state.horizon) - 1
    state.tick = 1

    return state.solution[0], {}

def step(state, action):
    assert state.tick < state.horizon
    assert action in (0, 1)

    ob = reward = 0.0

    if state.tick < state.mem_length:
        ob = state.solution[state.tick]
        reward = float(action == 0)

    if state.tick >= state.mem_length + state.mem_delay:
        idx = state.tick - state.mem_length - state.mem_delay
        sol = state.solution[idx]
        reward = float(action == sol)
        state.submission[state.tick] = action

    state.tick += 1
    terminal = state.tick == state.horizon

    info = {}
    if terminal:
        info['correct'] = np.all(
            state.solution[:state.mem_length] == state.submission[-state.mem_length:])

    return ob, reward, terminal, False, info

def render(state):
    def _render(val):
        if val == 1:
            c = 94
        elif val == 0:
            c = 91
        else:
            c = 90
        return f'\033[{c}m██\033[0m'

    chars = []
    for val in state.solution:
        c = _render(val)
        chars.append(c)
    chars.append(' Solution\n')

    for val in state.submission:
        c = _render(val)
        chars.append(c)
    chars.append(' Prediction\n')

    return ''.join(chars)

def close(state):
    pass

class Memory(gymnasium.Env):
    __init__ = init
    reset = reset
    step = step
    render = render
    close = close
