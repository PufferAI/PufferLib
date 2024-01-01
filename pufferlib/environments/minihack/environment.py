from pdb import set_trace as T

import gym
import shimmy

import pufferlib
import pufferlib.emulation
import pufferlib.environments


def env_creator():
    pufferlib.environments.try_import('minihack')
    return gym.make
 
def make_env(name='MiniHack-River-v0', render_mode='rgb_array'):
    '''NetHack binding creation function'''
    env = env_creator()(name, render_mode=render_mode)
    env = shimmy.GymV21CompatibilityV0(env=env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)
