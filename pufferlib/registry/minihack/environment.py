from pdb import set_trace as T

import gym
import shimmy

import pufferlib
import pufferlib.emulation
import pufferlib.registry


def env_creator():
    pufferlib.registry.try_import('minihack')
    return gym.make
 
def make_env(name='MiniHack-River-v0'):
    '''NetHack binding creation function'''
    env = env_creator()(name)
    env = shimmy.GymV21CompatibilityV0(env=env)
    return pufferlib.emulation.GymPufferEnv(env=env)
