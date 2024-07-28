from pdb import set_trace as T
import numpy as np
import functools

import gymnasium as gym

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.utils
import pufferlib.postprocess


def env_creator(name='doom'):
    return functools.partial(make, name)

def make(name, framestack=1, render_mode='rgb_array'):
    '''Atari creation function with default CleanRL preprocessing based on Stable Baselines3 wrappers'''
    if name == 'doom':
        name = 'VizdoomHealthGatheringSupreme-v0'

    #pufferlib.environments.try_import('vizdoom', 'gymnasium_wrapper')
    from stable_baselines3.common.atari_wrappers import (
        ClipRewardEnv,
        EpisodicLifeEnv,
        FireResetEnv,
        MaxAndSkipEnv,
        NoopResetEnv,
    )
    # Make does not work without this imported
    # TODO: Fix try_import
    from vizdoom import gymnasium_wrapper
    with pufferlib.utils.Suppress():
        env = gym.make(name, render_mode=render_mode)

    env = DoomWrapper(env) # Don't use standard postprocessor

    #env = gym.wrappers.RecordEpisodeStatistics(env)
    #env = NoopResetEnv(env, noop_max=30)
    #env = MaxAndSkipEnv(env, skip=4)
    #env = EpisodicLifeEnv(env)
    #if "FIRE" in env.unwrapped.get_action_meanings():
    #    env = FireResetEnv(env)
    #env = ClipRewardEnv(env)
    #env = gym.wrappers.GrayScaleObservation(env)
    #env = gym.wrappers.FrameStack(env, framestack)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

class DoomWrapper(gym.Wrapper):
    '''Gymnasium env does not expose proper options for screen scale and
    render format. This is slow. So we do it ourselves. Not it is fast. Yay!'''
    def __init__(self, env):
        super().__init__(env.unwrapped)
        if env.observation_space['screen'].shape[0] != 120:
            raise ValueError('Wrong screen resolution. Doom does not provide '
                'a way to change this. You must edit scenarios/<env_name>.cfg'
                'This is inside your local ViZDoom installation. Likely in python system packages'
                'Set screen resolution to RES_160X120 and screen format to GRAY8')

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(60, 80, 1), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs['screen'][::2, ::2], {}

    def step(self, action):
        obs, reward, terminal, truncated, info = self.env.step(action)
        return obs['screen'][::2, ::2], reward, terminal, truncated, info
