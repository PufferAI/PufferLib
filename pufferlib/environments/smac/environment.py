import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.wrappers


def env_creator(name='smac'):
    return functools.partial(make, name)

def make(name):
    '''Starcraft Multiagent Challenge creation function

    Support for SMAC is WIP because environments do not function without
    an action-masked baseline policy.'''
    pufferlib.environments.try_import('smac')
    from smac.env.pettingzoo.StarCraft2PZEnv import _parallel_env as smac_env

    env = smac_env(1000)
    env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
    env = pufferlib.emulation.PettingZooPufferEnv(env)
    return env
