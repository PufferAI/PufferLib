import gymnasium
import numpy as np
import random

from pufferlib import namespace


def init(self,
        mem_length=1,
        mem_delay=0,
        ):
    '''Pufferlib Memory environment

    Repeat the observed sequence after a delay. It is randomly generated upon every reset. This is a test of memory length and capacity. It starts requiring credit assignment if you make the sequence too long.

    The sequence is presented one digit at a time, followed by a string of 0. The agent should output 0s for the first mem_length + mem_delay steps, then output the sequence.

    Observation space: Box(0, 1, (1,)). The current digit.
    Action space: Discrete(2). Your guess for the next digit.

    Args:
        mem_length: The length of the sequence
        mem_delay: The number of 0s between the sequence and the agent's response
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
        info['score'] = np.all(
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
