from pdb import set_trace as T
import numpy as np

import pufferlib.emulation
import pufferlib.registry
import pufferlib.wrappers


def env_creator():
    pufferlib.registry.try_import('gym_microrts')
    from gym_microrts.envs import GlobalAgentCombinedRewardEnv
    return GlobalAgentCombinedRewardEnv

def make_env():
    '''Gym MicroRTS creation function
    
    This library appears broken. Step crashes in Java.
    '''
    env = env_creator()()
    env = MicroRTS(env)
    env = pufferlib.wrappers.GymToGymnasium(env)
    return pufferlib.emulation.GymPufferEnv(env=env)

class MicroRTS:
    def __init__(self, env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render = self.env.render
        self.close = self.env.close

    def reset(self):
        return self.env.reset().astype(np.int32)

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return o.astype(np.int32), r, d, i
