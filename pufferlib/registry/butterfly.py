from pettingzoo.utils.conversions import aec_to_parallel_wrapper

import pufferlib
import pufferlib.utils


def make_knights_archers_zombies_v10_binding():
    '''Knights Archers Zombies binding creation function'''
    try:
        from pettingzoo.butterfly import knights_archers_zombies_v10 as kaz
    except:
        raise pufferlib.utils.SetupError('Knights Archers Zombies v10')
    else:
        return pufferlib.emulation.Binding(
            env_cls=aec_to_parallel_wrapper,
            default_args=[kaz.raw_env()],
            env_name='kaz',
        )

def make_cooperative_pong_v5_binding():
    '''Cooperative Pong binding creation function'''
    try:
        from pettingzoo.butterfly import cooperative_pong_v5 as pong
    except:
        raise pufferlib.utils.SetupError('Cooperative Pong v5')
    else:
        return pufferlib.emulation.Binding(
            env_cls=aec_to_parallel_wrapper,
            default_args=[pong.raw_env()],
            env_name='cooperative-pong',
        )