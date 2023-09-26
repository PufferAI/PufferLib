from pdb import set_trace as T
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

import pufferlib.emulation
import pufferlib.exceptions


def env_creator(name):
    if name == 'battle_v4':
        try:
            from pettingzoo.magent import battle_v4 as battle
        except:
            raise pufferlib.exceptions.SetupError('magent', 'Battle V4')
    else:
        raise ValueError(f'Unknown environment name {name}')
    return battle.env
 
def make_env(name='battle_v4'):
    '''MAgent Battle V4 creation function'''
    env = env_creator(name)()
    env = aec_to_parallel_wrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)
