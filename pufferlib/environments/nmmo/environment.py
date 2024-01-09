from pdb import set_trace as T
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.wrappers


def env_creator(name='nmmo'):
    return functools.partial(make, name)

def make(name, *args, **kwargs):
    '''Neural MMO creation function'''
    nmmo = pufferlib.environments.try_import('nmmo')
    env = nmmo.Env(*args, **kwargs)
    env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)
