from pdb import set_trace as T

import gymnasium

from pokegym import PokemonRedV1 as env_creator

import pufferlib.emulation


def make_env():
    '''Pokemon Red'''
    env = env_creator()
    env = gymnasium.wrappers.ResizeObservation(env, shape=(72, 80))
    return pufferlib.emulation.GymnasiumPufferEnv(env=env,
        postprocessor_cls=pufferlib.emulation.BasicPostprocessor)
