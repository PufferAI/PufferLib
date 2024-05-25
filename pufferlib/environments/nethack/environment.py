from pdb import set_trace as T

import shimmy
import gym
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.postprocess
#from .wrapper import RenderCharImagesWithNumpyWrapper


def env_creator(name='NetHackScore-v0'):
    return functools.partial(make, name)

def make(name):
    '''NetHack binding creation function'''
    nle = pufferlib.environments.try_import('nle')
    env = gym.make(name)
    #env = RenderCharImagesWithNumpyWrapper(env)
    env = shimmy.GymV21CompatibilityV0(env=env)
    env = NethackWrapper(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

class NethackWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.close = self.env.close
        self.close = self.env.close
        self.render_mode = 'ansi'

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.obs = obs
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.obs = obs
        return obs, reward, done, truncated, info

    def render(self):
        import nle
        chars = nle.nethack.tty_render(
            self.obs['tty_chars'], self.obs['tty_colors'], self.obs['tty_cursor'])
        return chars
