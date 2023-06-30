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

def make_binding():
    '''Crafter binding creation function'''
    try:
        import crafter
        env_cls = lambda: gym.make('CrafterReward-v1')
        env_cls()
    except:
        raise pufferlib.utils.SetupError('Crafter')
    else:
        return pufferlib.emulation.Binding(
            env_cls=env_cls,
            env_name='Crafter',
            postprocessor_cls=CrafterPostprocessor,
            obs_dtype=np.uint8,
        )