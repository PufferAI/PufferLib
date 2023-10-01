import gym

import pufferlib
import pufferlib.emulation
import pufferlib.registry
import pufferlib.wrappers


def env_creator():
    pufferlib.registry.try_import('minerl')
    return gym.make

def make_env(name='MineRLNavigateDense-v0'):
    '''Minecraft environment creation function'''

    # Monkey patch to add .itmes to old gym.spaces.Dict
    gym.spaces.Dict.items = lambda self: self.spaces.items()

    env = env_creator()(name)
    env = pufferlib.wrappers.GymToGymnasium(env)
    return pufferlib.emulation.GymPufferEnv(env)
