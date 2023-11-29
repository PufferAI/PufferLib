from pdb import set_trace as T

import gym
import shimmy

import pufferlib
import pufferlib.emulation
import pufferlib.environments


def env_creator():
    pufferlib.environments.try_import('griddly')
    return gym.make

def make_env(name='GDY-Spiders-v0'):
    '''Griddly creation function

    Note that Griddly environments do not have observation spaces until
    they are created and reset'''
    with pufferlib.utils.Suppress():
        env = env_creator()(name)
        env.reset() # Populate observation space

    env = shimmy.GymV21CompatibilityV0(env=env)
    return pufferlib.emulation.GymnasiumPufferEnv(env)
