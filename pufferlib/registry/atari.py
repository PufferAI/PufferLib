from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.distributions.categorical import Categorical

import gym

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    #NoopResetEnv,
)

import pufferlib
import pufferlib.emulation
import pufferlib.models


Policy = pufferlib.models.Convolutional

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
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

class AtariFeaturizer(pufferlib.emulation.Postprocessor):
    def features(self, obs, step):
        return np.array(obs[1], dtype=np.float32)

    def infos(self, team_reward, team_done, team_infos, step):
        if 'lives' in team_infos:
            if team_infos['lives'] == 0 and team_done:
                team_infos['return'] = team_infos['episode']['r']
                team_infos['length'] = team_infos['episode']['l']
                team_infos['time'] = team_infos['episode']['t']
                return team_infos
            return {}

        if self.done:
            return

        if team_done:
            team_infos['return'] = self.epoch_return
            team_infos['length'] = self.epoch_length
            self.done = True
        else:
            self.epoch_length += 1
            self.epoch_return += team_reward

        return team_infos


def make_env(env_name, framestack):
    '''Atari creation function with default CleanRL preprocessing based on Stable Baselines3 wrappers'''
    try:
        with pufferlib.utils.Suppress():
            env = gym.make(env_name)
    except ImportError as e:
        raise e('Cannot gym.make ALE environment (pip install pufferlib[gym])')

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

    return env


def make_binding(env_name, framestack):
    '''Atari binding creation function'''
    try:
        make_env(env_name, framestack)
    except:
        raise pufferlib.utils.SetupError(f'{env_name} (ale)')
    else:
        return pufferlib.emulation.Binding(
            env_creator=make_env,
            default_args=[env_name, framestack],
            env_name=env_name,
            postprocessor_cls=AtariFeaturizer,
            emulate_flat_atn=True,
            suppress_env_prints=False,
        )