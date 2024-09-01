from pdb import set_trace as T

import gym
import shimmy
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.utils


def env_creator(name='MineRLBasaltFindCave-v0'):
    return functools.partial(make, name=name)

def make(name):
    '''Minecraft environment creation function'''

    pufferlib.environments.try_import('minerl')

    # Monkey patch to add .itmes to old gym.spaces.Dict
    #gym.spaces.Dict.items = lambda self: self.spaces.items()

    #with pufferlib.utils.Suppress():
    env = gym.make(name)

    env = shimmy.GymV21CompatibilityV0(env=env)
    return pufferlib.emulation.GymnasiumPufferEnv(env)
