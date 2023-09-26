import numpy as np

import gym

import pufferlib
import pufferlib.emulation
import pufferlib.exceptions


def env_creator():
    return gym.make

def make_env(name='GDY-Spiders-v0'):
    '''Griddly creation function

    Note that Griddly environments do not have observation spaces until
    they are created and reset'''
    try:
        import griddly
        with pufferlib.utils.Suppress():
            env = env_creator()(name)
    except:
        raise pufferlib.exceptions.SetupError('griddly', name)

    env.reset() # Populate observation space
    return pufferlib.emulation.GymPufferEnv(env)
