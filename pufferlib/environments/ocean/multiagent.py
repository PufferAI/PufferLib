import gymnasium
import pettingzoo
import numpy as np
import random

from pufferlib import namespace

def observation_space(state, agent):
    return gymnasium.spaces.Box(
        low=0, high=1, shape=(1,))

def action_space(state, agent):
    return gymnasium.spaces.Discrete(2)


def init(self):
    '''Pufferlib Multiagent environment

    Agent 1 must pick action 0 and Agent 2 must pick action 1

    Observation space: Box(0, 1, (1,)). 0 for Agent 1 and 1 for Agent 2
    Action space: Discrete(2). Which action to take.
    '''
    observation = {
        1: np.zeros(1, dtype=np.float32),
        2: np.ones(1, dtype=np.float32),
    }
    terminal = {
        1: True,
        2: True,
    }
    truncated = {
        1: False,
        2: False,
    }
    return namespace(self,
        observation=observation,
        terminal=terminal,
        truncated=truncated,
        possible_agents=[1, 2], 
        agents=[1, 2],
    )

def reset(state, seed=None):
    # Reallocating is faster than zeroing
    state.view=np.zeros((2, 5), dtype=np.float32)
    return state.observation, {}

def step(state, action):
    reward = {}
    assert 1 in action and action[1] in (0, 1)
    if action[1] == 0:
        state.view[0, 2] = 1
        reward[1] = 1
    else:
        state.view[0, 0] = 1
        reward[1] = 0

    assert 2 in action and action[2] in (0, 1)
    if action[2] == 1:
        state.view[1, 2] = 1
        reward[2] = 1
    else:
        state.view[1, 4] = 1
        reward[2] = 0

    info = {
        1: {'score': reward[1]},
        2: {'score': reward[2]},
    }
    return state.observation, reward, state.terminal, state.truncated, info

def render(state):
    def _render(val):
        if val == 1:
            c = 94
        elif val == 0:
            c = 90
        else:
            c = 90
        return f'\033[{c}m██\033[0m'

    chars = []
    for row in state.view:
        for val in row:
            c = _render(val)
            chars.append(c)
        chars.append('\n')
    return ''.join(chars)

def close(state):
    pass

class Multiagent(pettingzoo.ParallelEnv):
    __init__ = init
    reset = reset
    step = step
    render = render
    close = close
    observation_space = observation_space
    action_space = action_space
