from pdb import set_trace as T

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

def env_creator():
    try:
        import crafter
    except:
        raise pufferlib.exceptions.SetupError('crafter')
    else:
        return gym.make

def make_env(name='CrafterReward-v1'):
    '''Crafter creation function'''
    env = env_creator()(name)
    return pufferlib.emulation.GymPufferEnv(env=env)
