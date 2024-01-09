import gymnasium
import numpy as np
import random

from pufferlib import namespace


def init(self, password_length=5, hard_fixed_seed=42):
    '''Pufferlib Password environment

    Guess the password. This is a test for reward sparsity.
    '''
    return namespace(self,
        password_length=password_length,
        hard_fixed_seed=hard_fixed_seed,
        observation_space=gymnasium.spaces.Box(
            low=0, high=1, shape=(password_length,)),
        action_space=gymnasium.spaces.Discrete(2),
    )

def reset(state, seed=None):
    # Bandit problem requires a single fixed seed
    # for all environments
    seed = state.hard_fixed_seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    state.observation = np.zeros(state.password_length, dtype=np.float32) - 1
    state.solution = np.random.randint(
        0, 2, size=state.password_length).astype(np.float32)
    state.tick = 0

    return state.observation, {}

def step(state, action):
    assert state.tick < state.password_length
    assert action in (0, 1)

    state.observation[state.tick] = action
    state.tick += 1

    reward = 0
    terminal = state.tick == state.password_length
    info = {}

    if terminal:
        reward = float(np.all(state.observation == state.solution))
        info['correct'] = reward

    return state.observation, reward, terminal, False, info

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

    for val in state.observation:
        c = _render(val)
        chars.append(c)
    chars.append(' Prediction\n')

    return ''.join(chars)

def close(state):
    pass

class Password(gymnasium.Env):
    __init__ = init
    reset = reset
    step = step
    render = render
    close = close
