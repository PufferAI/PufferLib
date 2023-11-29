from pdb import set_trace as T
import numpy as np

import warnings

import shimmy

import pufferlib.emulation
import pufferlib.environments


def env_creator():
    pufferlib.environments.try_import('gym_microrts')
    from gym_microrts.envs import GlobalAgentCombinedRewardEnv
    return GlobalAgentCombinedRewardEnv

def make_env():
    '''Gym MicroRTS creation function
    
    This library appears broken. Step crashes in Java.
    '''
    with pufferlib.utils.Suppress():
        env = env_creator()()

    env.reset = pufferlib.utils.silence_warnings(env.reset)
    env.step = pufferlib.utils.silence_warnings(env.step)

    env = MicroRTS(env)
    env = shimmy.GymV21CompatibilityV0(env=env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

class MicroRTS:
    def __init__(self, env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render = self.env.render
        self.close = self.env.close
        self.seed = self.env.seed

    def reset(self):
        return self.env.reset().astype(np.int32)

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return o.astype(np.int32), r, d, i
