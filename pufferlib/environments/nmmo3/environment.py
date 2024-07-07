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

# 1024 params work here, but learning levels off
#def make(name, width=2048, height=2048, num_envs=1):
#    return PuffEnv(width=width, height=height, num_envs=num_envs)

def make(name, width=1024, height=1024, num_envs=1):
    return PuffEnv(width=width, height=height, num_envs=num_envs)

def make(name, num_envs=1):
    return PuffEnv(
        width=[1024, 1024, 1024, 1024],
        height=[256, 512, 1024, 2048],
        num_envs=4,
    )

# Testing with same env for now
def make(name, num_envs=1):
    return PuffEnv(
        width=8*[2048],
        height=8*[2048],
        num_players=8*[512],
        num_envs=8,
    )

# Testing with same env for now
def make(name, num_envs=1):
    return PuffEnv(
        width=2*[2048],
        height=2*[2048],
        num_envs=2,
    )


