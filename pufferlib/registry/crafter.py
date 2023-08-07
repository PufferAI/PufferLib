from pdb import set_trace as T
import numpy as np

import gym

import pufferlib
import pufferlib.emulation
import pufferlib.models


Policy = pufferlib.models.Convolutional

class CrafterPostprocessor(pufferlib.emulation.Postprocessor):
    def features(self, obs, step):
        return obs[1].transpose(2, 0, 1)

def make_env():
    '''Crafter creation function'''
    try:
        import crafter
    except:
        raise pufferlib.utils.SetupError('Crafter')
    else:
        env = gym.make('CrafterReward-v1')
        env = pufferlib.emulation.GymPufferEnv(env=env)
        return env