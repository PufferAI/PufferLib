from pdb import set_trace as T

import gymnasium
import functools

from nmmo3 import Environment, PuffEnv

import pufferlib.emulation
import pufferlib.postprocess


def env_creator(name='nmmo3'):
    return functools.partial(make, name)

def make(name, width=1024, height=1024, num_envs=1):
    env = Environment(width=width, height=height, num_envs=num_envs)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)

def make(name, width=1024, height=1024, num_envs=1):
    return PuffEnv(width=width, height=height, num_envs=num_envs)
