from pdb import set_trace as T
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
import functools

import pufferlib.emulation
import pufferlib.environments
import pufferlib.wrappers


def env_creator(name='battle_v4'):
    return functools.partial(make, name)
    pufferlib.environments.try_import('pettingzoo.magent', 'magent')

def make(name):
    '''MAgent Battle V4 creation function'''
    if name == 'battle_v4':
        from pettingzoo.magent import battle_v4
        env_cls = battle_v4.env
    else:
        raise ValueError(f'Unknown environment name {name}')
 
    env = env_cls()
    env = aec_to_parallel_wrapper(env)
    env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)
