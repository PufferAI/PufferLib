from pdb import set_trace as T
import numpy as np

import gym

import pufferlib
import pufferlib.emulation
import pufferlib.exceptions
import pufferlib.models


class Policy(pufferlib.models.Convolutional):
    def __init__(
            self,
            envs,
            flat_size=1024,
            channels_last=True,
            downsample=1,
            input_size=512,
            hidden_size=128,
            output_size=128,
            **kwargs
        ):
        super().__init__(
            envs,
            framestack=3,
            flat_size=flat_size,
            channels_last=channels_last,
            downsample=downsample,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            **kwargs
        )


class CrafterPostprocessor(pufferlib.emulation.Postprocessor):
    def features(self, obs, step):
        return obs[1].transpose(2, 0, 1)


def make_env():
    '''Crafter creation function'''
    try:
        import crafter
    except:
        raise pufferlib.exceptions.SetupError('crafter', 'CrafterReward-v1')
    else:
        env = gym.make('CrafterReward-v1')
        env = pufferlib.emulation.GymPufferEnv(env=env)
        return env