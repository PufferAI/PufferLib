from pdb import set_trace as T

import gym
import shimmy
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments


def env_creator(name='GDY-Spiders-v0'):
    return functools.partial(make, name)

def make(name):
    '''Griddly creation function

    Note that Griddly environments do not have observation spaces until
    they are created and reset'''
    pufferlib.environments.try_import('griddly')
    with pufferlib.utils.Suppress():
        env = gym.make(name)
        env.reset() # Populate observation space

    env = shimmy.GymV21CompatibilityV0(env=env)
    return pufferlib.emulation.GymnasiumPufferEnv(env)
