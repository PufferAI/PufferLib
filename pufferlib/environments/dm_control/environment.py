from pdb import set_trace as T

import gym
import shimmy

import pufferlib
import pufferlib.emulation
import pufferlib.environments


def env_creator():
    '''Deepmind Control environment creation function

    No support for bindings yet because PufferLib does
    not support continuous action spaces.'''
    dm_control = pufferlib.environments.try_import('dm_control.suite', 'dmc')
    return dm_control.suite.load

def make_env(domain_name='walker', task_name='walk'):
    '''No PufferLib support for continuous actions yet.'''
    env = env_creator()(domain_name, task_name)
    env = shimmy.DmControlCompatibilityV0(env=env)
    return pufferlib.emulation.GymnasiumPufferEnv(env)
