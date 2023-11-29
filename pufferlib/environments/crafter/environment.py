from pdb import set_trace as T

import gym
import shimmy

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.utils


class CrafterPostprocessor(pufferlib.emulation.Postprocessor):
    def features(self, obs, step):
        return obs[1].transpose(2, 0, 1)

def env_creator():
    pufferlib.environments.try_import('crafter')
    return gym.make

def make_env(name='CrafterReward-v1'):
    '''Crafter creation function'''
    env = pufferlib.utils.silence_warnings(env_creator())(name)
    env.reset = pufferlib.utils.silence_warnings(env.reset)
    env = shimmy.GymV21CompatibilityV0(env=env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)
