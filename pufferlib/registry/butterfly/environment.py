from pdb import set_trace as T
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

import pufferlib.emulation
import pufferlib.exceptions


def env_creator(name):
    if name == 'cooperative_pong_v5':
        try:
            from pettingzoo.butterfly import cooperative_pong_v5 as pong
            return pong.raw_env
        except:
            raise pufferlib.exceptions.SetupError('butterfly', 'Cooperative Pong v5')
    elif name == 'knights_archers_zombies_v10':
        try:
            from pettingzoo.butterfly import knights_archers_zombies_v10 as kaz
            return kaz.raw_env
        except:
            raise pufferlib.exceptions.SetupError('butterfly', 'Knights Archers Zombies v10')
    else:
        raise ValueError(f'Unknown environment: {name}')
     
def make_env(name='cooperative_pong_v5'):
    env = env_creator(name)()
    env = aec_to_parallel_wrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)
