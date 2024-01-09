from pdb import set_trace as T

import gym
import shimmy
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.utils


class CrafterPostprocessor(pufferlib.emulation.Postprocessor):
    def features(self, obs, step):
        return obs[1].transpose(2, 0, 1)

def env_creator(name='CrafterReward-v1'):
    return functools.partial(make, name)

def make(name):
    '''Crafter creation function'''
    pufferlib.environments.try_import('crafter')
    env = gym.make(name)
    env.reset = pufferlib.utils.silence_warnings(env.reset)
    env = shimmy.GymV21CompatibilityV0(env=env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)
