from pdb import set_trace as T
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
import functools

import pufferlib.emulation
import pufferlib.environments


def env_creator(name='cooperative_pong_v5'):
    return functools.partial(make, name)

def make(name, render_mode='not_human'):
    pufferlib.environments.try_import('pettingzoo.butterfly', 'butterfly')
    if name == 'cooperative_pong_v5':
        from pettingzoo.butterfly import cooperative_pong_v5 as pong
        env_cls = pong.raw_env
    elif name == 'knights_archers_zombies_v10':
        from pettingzoo.butterfly import knights_archers_zombies_v10 as kaz
        env_cls = kaz.raw_env
    else:
        raise ValueError(f'Unknown environment: {name}')

    env = env_cls(render_mode=render_mode)
    env = aec_to_parallel_wrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)
