from pdb import set_trace as T
import numpy as np
import functools

import gymnasium as gym

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.utils
import pufferlib.postprocess


def env_creator(name='BreakoutNoFrameskip-v4'):
    return functools.partial(make, name)

def make(name, framestack=4, render_mode='rgb_array'):
    '''Atari creation function with default CleanRL preprocessing based on Stable Baselines3 wrappers'''
    pufferlib.environments.try_import('ale_py', 'atari')
    from stable_baselines3.common.atari_wrappers import (
        ClipRewardEnv,
        EpisodicLifeEnv,
        FireResetEnv,
        MaxAndSkipEnv,
        NoopResetEnv,
    )
    with pufferlib.utils.Suppress():
        env = gym.make(name, render_mode=render_mode)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, framestack)
    env = AtariPostprocessor(env) # Don't use standard postprocessor
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

class AtariPostprocessor(gym.Wrapper):
    '''Atari breaks the normal PufferLib postprocessor because
    it sends terminal=True every live, not every episode'''
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs, {}

    def step(self, action):
        obs, reward, terminal, truncated, info = self.env.step(action)
        if 'episode' not in info:
            info = {}

        return obs, reward, terminal, truncated, info


