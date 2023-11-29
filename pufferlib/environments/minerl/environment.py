from pdb import set_trace as T

import gym
import shimmy

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.utils


def env_creator():
    pufferlib.environments.try_import('minerl')
    return gym.make

def make_env(name='MineRLNavigateDense-v0'):
    '''Minecraft environment creation function'''

    # Monkey patch to add .itmes to old gym.spaces.Dict
    gym.spaces.Dict.items = lambda self: self.spaces.items()
    with pufferlib.utils.Suppress():
        env = env_creator()(name)

    env = shimmy.GymV21CompatibilityV0(env=env)
    return pufferlib.emulation.GymnasiumPufferEnv(env)
