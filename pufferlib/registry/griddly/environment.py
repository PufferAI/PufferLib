import numpy as np

import gym

import pufferlib
import pufferlib.emulation
import pufferlib.registry
import pufferlib.wrappers


def env_creator():
    pufferlib.registry.try_import('griddly')
    return gym.make

def make_env(name='GDY-Spiders-v0'):
    '''Griddly creation function

    Note that Griddly environments do not have observation spaces until
    they are created and reset'''
    with pufferlib.utils.Suppress():
        env = env_creator()(name)

    env.reset() # Populate observation space
    env = pufferlib.wrappers.GymToGymnasium(env)
    return pufferlib.emulation.GymPufferEnv(env)
