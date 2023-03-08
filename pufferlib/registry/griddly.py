import numpy as np

import gym

import pufferlib
import pufferlib.emulation


def make_spider_v0_binding():
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