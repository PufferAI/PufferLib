import gymnasium
import numpy as np
import random
import time

from pufferlib import namespace


def init(self, delay_mean=0, delay_std=0, bandwidth=1):
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
    start = time.process_time()
    idx = 0
    target_time = state.delay_mean + state.delay_std*np.random.randn()
    while time.process_time() - start < target_time:
        idx += 1

    return state.observation, 0, False, False, {}

def close(state):
    pass

def render(state):
    pass

class Performance(gymnasium.Env):
    __init__ = init
    reset = reset
    step = step
    render = render
    close = close
