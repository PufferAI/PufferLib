from pdb import set_trace as T
import numpy as np
import functools

import gymnasium as gym

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.utils


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
    return pufferlib.emulation.GymnasiumPufferEnv(
        env=env, postprocessor_cls=AtariFeaturizer)

# Broken in SB3
class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: the environment to wrap
    :param noop_max: the maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        for _ in range(noops):
            obs, _, done, _, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs, {}

class AtariFeaturizer(pufferlib.emulation.Postprocessor):
    def reset(self, obs):
        self.epoch_return = 0
        self.epoch_length = 0
        self.done = False

    #@property
    #def observation_space(self):
    #    return gym.spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)

    def observation(self, obs):
        return np.array(obs)
        return np.array(obs[1], dtype=np.float32)

    def reward_done_truncated_info(self, reward, done, truncated, info):
        return reward, done, truncated, info
        if 'lives' in info:
            if info['lives'] == 0 and done:
                info['return'] = info['episode']['r']
                info['length'] = info['episode']['l']
                info['time'] = info['episode']['t']
                return reward, True, info
            return reward, False, info

        if self.done:
            return reward, done, info

        if done:
            info['return'] = self.epoch_return
            info['length'] = self.epoch_length
            self.done = True
        else:
            self.epoch_length += 1
            self.epoch_return += reward

        return reward, done, info
