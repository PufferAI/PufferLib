import gymnasium
import numpy as np
import random

from pufferlib import namespace


def init(self):
    '''Pufferlib Spaces environment

    A simple environment with hierarchical observation and action spaces

    The image action should be 1 if the sum of the image is positive, 0 otherwise
    The flat action should be 1 if the sum of the flat obs is positive, 0 otherwise

    0.5 reward is given for each correct action

    Does not provide rendering
    '''

    observation_space = gymnasium.spaces.Dict({
        'image': gymnasium.spaces.Box(
            low=0, high=1, shape=(5, 5), dtype=np.float32),
        'flat': gymnasium.spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32),
    })
    action_space = gymnasium.spaces.Dict({
        'image': gymnasium.spaces.Discrete(2),
        'flat': gymnasium.spaces.Discrete(2),
    })

    return namespace(**locals())

def reset(state, seed=None):
    state.observation = {
        'image': np.random.randn(5, 5).astype(np.float32),
        'flat': np.random.randn(5).astype(np.float32),
    }
    state.image_sign = np.sum(state.observation['image']) > 0
    state.flat_sign = np.sum(state.observation['flat']) > 0

    return state.observation, {}

def step(state, action):
    assert isinstance(action, dict)
    assert 'image' in action and action['image'] in (0, 1)
    assert 'flat' in action and action['flat'] in (0, 1)

    reward = 0
    if state.image_sign == action['image']:
        reward += 0.5

    if state.flat_sign == action['flat']:
        reward += 0.5

    info = dict(score=reward)
    return state.observation, reward, True, False, info

def render(state):
    return ''

def close(state):
    pass

class Spaces(gymnasium.Env):
    __init__ = init
    reset = reset
    step = step
    render = render
    close = close
