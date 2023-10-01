import gym

import pufferlib
import pufferlib.emulation
import pufferlib.registry


def env_creator():
    '''Deepmind Control environment creation function

    No support for bindings yet because PufferLib does
    not support continuous action spaces.'''
    pufferlib.registry.try_import('dm_control', 'dmc')
    pufferlib.registry.try_import('gym_dmc', 'dmc')
    return gym.make

def make_env(name, *args):
    '''No PufferLib support for Deepmind Control environments yet.'''
    env = env_creator()(name, *args)
    return env
