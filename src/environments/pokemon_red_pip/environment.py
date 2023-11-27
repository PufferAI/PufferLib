from pdb import set_trace as T

import gymnasium

from pokegym import Environment as env_creator

import pufferlib.emulation


def make_env(headless: bool = True, state_path=None):
    '''Pokemon Red'''
    env = env_creator(headless=headless, state_path=state_path)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env,
        postprocessor_cls=pufferlib.emulation.BasicPostprocessor)
