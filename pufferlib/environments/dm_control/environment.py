from pdb import set_trace as T

import gym
import shimmy
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments


def env_creator(name='walker'):
    '''Deepmind Control environment creation function

    No support for bindings yet because PufferLib does
    not support continuous action spaces.'''
    return functools.partial(make, name)

def make(name, task_name='walk'):
    '''No PufferLib support for continuous actions yet.'''
    dm_control = pufferlib.environments.try_import('dm_control.suite', 'dmc')
    env = dm_control.suite.load(name, task_name)
    env = shimmy.DmControlCompatibilityV0(env=env)
    return pufferlib.emulation.GymnasiumPufferEnv(env)
