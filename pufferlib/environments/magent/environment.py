from pdb import set_trace as T
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

import pufferlib.emulation
import pufferlib.environments
import pufferlib.wrappers


def env_creator(name):
    pufferlib.environments.try_import('pettingzoo.magent', 'magent')
    if name == 'battle_v4':
        from pettingzoo.magent import battle_v4
    else:
        raise ValueError(f'Unknown environment name {name}')
    return battle_v4.env
 
def make_env(name='battle_v4'):
    '''MAgent Battle V4 creation function'''
    env = env_creator(name)()
    env = aec_to_parallel_wrapper(env)
    env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)
