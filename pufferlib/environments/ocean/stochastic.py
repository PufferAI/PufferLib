import gymnasium
import numpy as np
import random

from pufferlib import namespace


def init(self,
        p=0.75,
        horizon=1000,
        ):
    '''Pufferlib Stochastic environment

    Rewarded for playing action 0 < p % of the time and action 1 < (1 - p) %
    '''
    return namespace(self,
        p=p,
        horizon=horizon,
        observation=np.zeros(1, dtype=np.float32),
        observation_space=gymnasium.spaces.Box(
            low=0, high=1, shape=(1,)),
        action_space=gymnasium.spaces.Discrete(2),
    )

def reset(state, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    state.tick = 0
    state.count = 0
    state.action = 0

    return state.observation, {}

def step(state, action):
    assert state.tick < state.horizon
    assert action in (0, 1)

    state.tick += 1
    state.count += action == 0
    state.action = action

    terminal = state.tick == state.horizon
    atn0_frac = state.count / state.tick
    proximity_to_p = 1 - (state.p - atn0_frac)**2

    reward = proximity_to_p if (
        (action == 0 and atn0_frac < state.p) or
        (action == 1 and atn0_frac >= state.p)) else 0

    info = {}
    if terminal:
        info['score'] = proximity_to_p

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
    solution = 0 if state.count / state.tick < state.p else 1
    chars.append(_render(solution))
    chars.append(' Solution\n')

    chars.append(_render(state.action))
    chars.append(' Prediction\n')

    return ''.join(chars)

def close(state):
    pass

class Stochastic(gymnasium.Env):
    __init__ = init
    reset = reset
    step = step
    render = render
    close = close
