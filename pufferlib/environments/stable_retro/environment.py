from pdb import set_trace as T
import numpy as np

import gymnasium as gym

import pufferlib
import pufferlib.emulation
import pufferlib.environments


def env_creator():
    retro = pufferlib.environments.try_import('retro', 'stable-retro')
    return retro.make

def make_env(name='Airstriker-Genesis', framestack=4):
    '''Atari creation function with default CleanRL preprocessing based on Stable Baselines3 wrappers'''
    try:
        from stable_baselines3.common.atari_wrappers import (
            ClipRewardEnv,
            EpisodicLifeEnv,
            FireResetEnv,
            MaxAndSkipEnv,
        )
        with pufferlib.utils.Suppress():
            env = env_creator()(name)
    except Exception as e:
        raise e

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, framestack)
    return pufferlib.emulation.GymnasiumPufferEnv(
        env=env, postprocessor_cls=AtariFeaturizer)

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
