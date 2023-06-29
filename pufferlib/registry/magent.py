from pettingzoo.utils.conversions import aec_to_parallel_wrapper

import pufferlib.emulation
import pufferlib.models
import pufferlib.utils

Policy = pufferlib.models.Default

def make_battle_v4_binding():
    '''MAgent Battle binding creation function'''
    try:
        from pettingzoo.magent import battle_v4 as battle
    except:
        raise pufferlib.utils.SetupError('MAgent (pettingzoo)')
    else:
        return pufferlib.emulation.Binding(
            env_cls=aec_to_parallel_wrapper,
            default_args=[battle.env()],
            env_name='MAgent Battle v4',
        )