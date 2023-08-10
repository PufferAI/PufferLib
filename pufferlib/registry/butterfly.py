from pdb import set_trace as T
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

import pufferlib.emulation
import pufferlib.exceptions
import pufferlib.models


class Policy(pufferlib.models.Convolutional):
    def __init__(
            self,
            envs,
            flat_size=3520,
            channels_last=True,
            downsample=4,
            input_size=512,
            hidden_size=128,
            output_size=128,
            **kwargs
        ):
        super().__init__(
            envs,
            framestack=3,
            flat_size=flat_size,
            channels_last=channels_last,
            downsample=downsample,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            **kwargs
        )


def make_knights_archers_zombies_v10():
    '''Knights Archers Zombies creation function
    
    Not yet supported: requires heterogeneous observations''
    '''
    try:
        from pettingzoo.butterfly import knights_archers_zombies_v10 as kaz
    except:
        raise pufferlib.exceptions.SetupError('butterfly', 'Knights Archers Zombies v10')
    else:
        return pufferlib.emulation.PettingZooPufferEnv(
            env_creator=aec_to_parallel_wrapper,
            env_args=[kaz.raw_env()],
        )


def make_cooperative_pong_v5():
    '''Cooperative Pong creation function'''
    try:
        from pettingzoo.butterfly import cooperative_pong_v5 as pong
    except:
        raise pufferlib.exceptions.SetupError('butterfly', 'Cooperative Pong v5')
    else:
        return pufferlib.emulation.PettingZooPufferEnv(
            env_creator=aec_to_parallel_wrapper,
            env_args=[pong.raw_env()],
        )