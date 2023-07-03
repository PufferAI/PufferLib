import numpy as np

import gym

import pufferlib
import pufferlib.emulation

Policy = pufferlib.models.Default

def make_spider_v0_binding():
    '''Griddly Spiders binding creation function

    Support for Griddly is WIP because environments do not specify
    their observation spaces until after they are created.'''
    try:
        import griddly
        env_cls = lambda: gym.make('GDY-Spiders-v0')
        env_cls()
    except:
        raise pufferlib.utils.SetupError('Spiders-v0 (griddly)')
    else:
        return pufferlib.emulation.Binding(
            env_cls=env_cls,
            obs_dtype=np.uint8,
        )