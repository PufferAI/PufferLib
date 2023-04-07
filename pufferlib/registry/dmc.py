import gym

import pufferlib
import pufferlib.emulation
import pufferlib.utils

def make(name, *args):
    '''Deepmind Control environment creation function

    No support for bindings yet because PufferLib does
    not support continuous action spaces.'''
    try:
        from dm_control import suite
        import gym_dmc
    except:
        raise pufferlib.utils.SetupError('Deepmind Control (dmc)')
    else:
        return gym.make(name, *args)