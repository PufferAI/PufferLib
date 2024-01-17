from pdb import set_trace as T
import numpy as np

import warnings
import shimmy
import functools

import pufferlib.emulation
import pufferlib.environments


def env_creator(name='GlobalAgentCombinedRewardEnv'):
    return functools.partial(make, name)

def make(name):
    '''Gym MicroRTS creation function
    
    This library appears broken. Step crashes in Java.
    '''
    pufferlib.environments.try_import('gym_microrts')
    if name == 'GlobalAgentCombinedRewardEnv':
        from gym_microrts.envs import GlobalAgentCombinedRewardEnv
    else:
        raise ValueError(f'Unknown environment: {name}')

    with pufferlib.utils.Suppress():
        return GlobalAgentCombinedRewardEnv()

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
