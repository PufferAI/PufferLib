import numpy as np

import gym

import pufferlib
import pufferlib.emulation


def make_binding():
    '''Crafter binding creation function'''
    try:
        import crafter
        env_cls = lambda: gym.make('CrafterReward-v1')
        env_cls()
    except:
        raise pufferlib.utils.SetupError('Crafter')
    else:
        return pufferlib.emulation.Binding(
            env_cls=env_cls,
            env_name='Crafter',
            obs_dtype=np.uint8,
        )