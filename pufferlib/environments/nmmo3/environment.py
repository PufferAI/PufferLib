from pdb import set_trace as T

import gymnasium
import functools

from nmmo3 import Environment

import pufferlib.emulation


def env_creator(name='nmmo3'):
    return functools.partial(make, name)

def make(name):
    env = Environment()
    return pufferlib.emulation.GymnasiumPufferEnv(env=env,
        postprocessor_cls=pufferlib.emulation.BasicPostprocessor)
