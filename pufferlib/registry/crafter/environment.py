from pdb import set_trace as T

import gym

import pufferlib
import pufferlib.emulation
import pufferlib.exceptions


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
