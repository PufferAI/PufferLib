import gym

import pufferlib
import pufferlib.emulation
import pufferlib.exceptions

def env_creator(name):
    '''Deepmind Control environment creation function

    No support for bindings yet because PufferLib does
    not support continuous action spaces.'''
    try:
        from dm_control import suite
        import gym_dmc
    except:
        raise pufferlib.exceptions.SetupError('dmc', name)
    else:
        return gym.make

def make_env(name *args):
    '''No PufferLib support for Deepmind Control environments yet.'''
    env = env_creator(name)(*args)
    return env
