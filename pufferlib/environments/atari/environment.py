from pdb import set_trace as T
import numpy as np
import functools

import gymnasium as gym

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.utils
import pufferlib.postprocess
import pufferlib.postprocess

def env_creator(name='breakout'):
    return functools.partial(make, name)

def make(name, obs_type='grayscale', frameskip=4,
        repeat_action_probability=0.25, render_mode='rgb_array'):
    '''Atari creation function'''
    pufferlib.environments.try_import('ale_py', 'AtariEnv')
    from ale_py import AtariEnv
    env = AtariEnv(name, obs_type='grayscale', frameskip=frameskip,
        repeat_action_probability=repeat_action_probability,
        render_mode=render_mode)
                    
    env = pufferlib.postprocess.ResizeObservation(env, downscale=2)
    env = AtariPostprocessor(env) # Don't use standard postprocessor
    env = pufferlib.postprocess.EpisodeStats(env)
    env = pufferlib.emulation.GymnasiumPufferEnv(env=env)
    return env

class AtariPostprocessor(gym.Wrapper):
    '''Atari breaks the normal PufferLib postprocessor because
    it sends terminal=True every live, not every episode'''
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, *env.observation_space.shape),
            dtype=env.observation_space.dtype)

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        return np.expand_dims(obs, 0), {}

    def step(self, action):
        obs, reward, terminal, truncated, _ = self.env.step(action)
        return np.expand_dims(obs, 0), reward, terminal, truncated, {}

