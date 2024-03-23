from pdb import set_trace as T

import gymnasium
import functools

from nmmo3 import Environment

import pufferlib.emulation


def env_creator(name='nmmo3'):
    return functools.partial(make, name)

def make(name, width=256, height=256, num_envs=1):
    env = Environment(width=width, height=height, num_envs=num_envs)
    return pufferlib.emulation.PettingZooPufferEnv(env=env,
        postprocessor_cls=pufferlib.emulation.BasicPostprocessor)
