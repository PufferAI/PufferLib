
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

def env_creator(name='crafter'):
    return functools.partial(make, name)

def make(name, buf=None):
    '''Crafter creation function'''
    if name == 'crafter':
        name = 'CrafterReward-v1'

    pufferlib.environments.try_import('crafter')
    env = gym.make(name)
    env.reset = pufferlib.utils.silence_warnings(env.reset)
    env = shimmy.GymV21CompatibilityV0(env=env)
    env = RenderWrapper(env)
    env = TransposeObs(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

class RenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    @property
    def render_mode(self):
        return 'rgb_array'

    def render(self, *args, **kwargs):
        return self.env.unwrapped.env.unwrapped.render((256,256))
