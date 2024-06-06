from pdb import set_trace as T

import gym
import gymnasium
import shimmy
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.postprocess
import pufferlib.utils


class TransposeObs(gym.Wrapper):
    def observation(self, observation):
        return observation.transpose(2, 0, 1)

def env_creator(name='CrafterReward-v1'):
    return functools.partial(make, name)

def make(name):
    '''Crafter creation function'''
    pufferlib.environments.try_import('crafter')
    env = gym.make(name)
    env.reset = pufferlib.utils.silence_warnings(env.reset)
    env = shimmy.GymV21CompatibilityV0(env=env)
    env = RenderWrapper(env)
    env = TransposeObs(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

class RenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    @property
    def render_mode(self):
        return 'rgb_array'

    def render(self, *args, **kwargs):
        return self.env.unwrapped.env.unwrapped.render((256,256))
