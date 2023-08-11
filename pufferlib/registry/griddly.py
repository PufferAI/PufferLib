import numpy as np

import gym

import pufferlib
import pufferlib.emulation
import pufferlib.exceptions

Policy = pufferlib.models.Default

class GriddlyGymPufferEnv(pufferlib.emulation.GymPufferEnv):
    def __init__(self, env_creator, env_args=[], env_kwargs={}):
        '''Griddly envs need to be reset in order to define their obs space'''
        def reset_env_creator(*args, **kwargs):
            env = env_creator(*args, **kwargs)
            env.reset()
            return env

        super().__init__(
            env_creator=reset_env_creator,
            env_args=env_args,
            env_kwargs=env_kwargs,
        )

def make_spider_v0_env():
    '''Griddly Spiders binding creation function

    Support for Griddly is WIP because environments do not specify
    their observation spaces until after they are created.'''
    try:
        import griddly
        with pufferlib.utils.Suppress():
            env_cls = lambda: gym.make('GDY-Spiders-v0')
            env_cls()
    except:
        raise pufferlib.exceptions.SetupError('griddly', 'GDY-Spiders-v0')
    else:
        return GriddlyGymPufferEnv(
            env_creator=env_cls,
        )