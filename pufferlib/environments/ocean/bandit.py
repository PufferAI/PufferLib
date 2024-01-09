import gymnasium
import numpy as np
import random

from pufferlib import namespace


def init(self,
        num_actions=4,
        reward_scale=1,
        reward_noise=0,
        hard_fixed_seed=42,
        ):
    '''Pufferlib Bandit environment

    Simulates a variety of classic bandit problems
    '''
    return namespace(self,
        num_actions=num_actions,
        reward_scale=reward_scale,
        reward_noise=reward_noise,
        hard_fixed_seed=hard_fixed_seed,
        observation=np.ones(1, dtype=np.float32),
        observation_space=gymnasium.spaces.Box(
            low=-1, high=1, shape=(1,)),
        action_space=gymnasium.spaces.Discrete(num_actions),
    )

def reset(state, seed=None):
    # Bandit problem requires a single fixed seed
    # for all environments
    seed = state.hard_fixed_seed

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    state.solution_idx = np.random.randint(0, state.num_actions)

    return state.observation, {}

def step(state, action):
    assert action == int(action) and action >= 0 and action < state.num_actions

    correct = False
    reward = 0
    if action == state.solution_idx:
        correct = True
        reward = 1

    reward_noise = 0
    if state.reward_noise != 0:
        reward_noise = np.random.randn() * state.reward_scale

    # Couples reward noise to scale
    reward = (reward + reward_noise) * state.reward_scale

    return state.observation, reward, True, False, {'correct': correct}

def render(state):
    pass

def close(state):
    pass

class Bandit(gymnasium.Env):
    __init__ = init
    reset = reset
    step = step
    render = render
    close = close
