from pdb import set_trace as T

import gym
import shimmy
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments


def env_creator(name='seekavoid_arena_01'):
    '''Deepmind Lab binding creation function
    dm-lab requires extensive setup. Use PufferTank.'''
    return functools.partial(make, name=name)

def make(name):
    '''Deepmind Lab binding creation function
    dm-lab requires extensive setup. Use PufferTank.'''
    dm_lab = pufferlib.environments.try_import('deepmind_lab', 'dm-lab')
    env = dm_lab.Lab(name, ['RGB_INTERLEAVED'])
    env = shimmy.DmLabCompatibilityV0(env=env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)
