from pdb import set_trace as T

import gymnasium
import functools

from pokegym import Environment

import pufferlib.emulation


def env_creator(name='pokemon_red'):
    return functools.partial(make, name)

def make(name, headless: bool = True, state_path=None):
    '''Pokemon Red'''
    env = Environment(headless=headless, state_path=state_path)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env,
        postprocessor_cls=pufferlib.emulation.BasicPostprocessor)
