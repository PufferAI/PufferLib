from pdb import set_trace as T

import gym
import shimmy

import pufferlib
import pufferlib.emulation
import pufferlib.environments


def env_creator():
    '''Deepmind Lab binding creation function
    dm-lab requires extensive setup. Use PufferTank.'''
    dm_lab = pufferlib.environments.try_import('deepmind_lab', 'dm-lab')
    return dm_lab.Lab

def make_env(name='seekavoid_arena_01'):
    '''Deepmind Lab binding creation function
    dm-lab requires extensive setup. Use PufferTank.'''
    env = env_creator()(name, ['RGB_INTERLEAVED'])
    env = shimmy.DmLabCompatibilityV0(env=env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)
