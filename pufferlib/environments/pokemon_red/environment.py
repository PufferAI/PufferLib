from pdb import set_trace as T

import gymnasium
import functools

from pokegym import Environment

import pufferlib.emulation
import pufferlib.postprocess


def env_creator(name='pokemon_red'):
    return functools.partial(make, name)

def make(name, headless: bool = True, state_path=None):
    '''Pokemon Red'''
    env = Environment(headless=headless, state_path=state_path)
    env.render_mode = 'rgb_array'
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)
