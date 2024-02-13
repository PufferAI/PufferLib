import gymnasium
import numpy as np
import random
import time

from pufferlib import namespace


def init(self, count_n=0, count_std=0, bandwidth=1):
    np.random.seed(time.time_ns() % 2**32)

    observation_space = gymnasium.spaces.Box(
        low=-2**20, high=2**20,
        shape=(bandwidth,), dtype=np.float32
    )
    action_space = gymnasium.spaces.Discrete(2)
    observation = observation_space.sample()
    return namespace(**locals())


def reset(state, seed=None):
    return state.observation, {}

def step(state, action):
    idx = 0
    target = state.count_n  +  state.count_std * np.random.randn()
    while idx < target:
        idx += 1

    return state.observation, 0, False, False, {}

def close(state):
    pass

def render(state):
    pass

class PerformanceEmpiric(gymnasium.Env):
    __init__ = init
    reset = reset
    step = step
    render = render
    close = close
