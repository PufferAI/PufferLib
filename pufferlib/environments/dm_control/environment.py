
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

def make(name, task_name='walk', buf=None):
    '''Untested. Let us know in Discord if you want to use dmc in PufferLib.'''
    dm_control = pufferlib.environments.try_import('dm_control.suite', 'dmc')
    env = dm_control.suite.load(name, task_name)
    env = shimmy.DmControlCompatibilityV0(env=env)
    return pufferlib.emulation.GymnasiumPufferEnv(env, buf=buf)
