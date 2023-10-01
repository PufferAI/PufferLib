from pdb import set_trace as T

import gym

import pufferlib
import pufferlib.emulation
import pufferlib.registry
import pufferlib.wrappers


class CrafterPostprocessor(pufferlib.emulation.Postprocessor):
    def features(self, obs, step):
        return obs[1].transpose(2, 0, 1)

def env_creator():
    pufferlib.registry.try_import('crafter')
    return gym.make

def make_env(name='CrafterReward-v1'):
    '''Crafter creation function'''
    env = env_creator()(name)
    env = pufferlib.wrappers.GymToGymnasium(env)
    return pufferlib.emulation.GymPufferEnv(env=env)
