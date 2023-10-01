import gym

import pufferlib
import pufferlib.emulation
import pufferlib.registry
import pufferlib.wrappers


def env_creator():
    '''Deepmind Lab binding creation function
    dm-lab requires extensive setup. Use PufferTank.'''
    pufferlib.registry.try_import('gym_deepmindlab', 'dm-lab')
    return gym.make

def make_env(name='DeepmindLabSeekavoidArena01-v0'):
    '''Deepmind Lab binding creation function
    dm-lab requires extensive setup. Use PufferTank.'''
    env = env_creator()(name)
    env = pufferlib.wrappers.GymToGymnasium(env)
    return pufferlib.emulation.GymPufferEnv(env=env)
