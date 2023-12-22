from pdb import set_trace as T

import shimmy
import gym

import pufferlib
import pufferlib.emulation
import pufferlib.environments


def env_creator():
    nle = pufferlib.environments.try_import('nle')
    return gym.make
    return nle.env.NLE
 
def make_env(name='NetHackScore-v0'):
    '''NetHack binding creation function'''
    env = env_creator()(name)
    env = shimmy.GymV21CompatibilityV0(env=env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)
