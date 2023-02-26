from pdb import set_trace as T

import random

import numpy as np

import gym

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    #NoopResetEnv,
)

import pufferlib
import pufferlib.registry


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

def make_env():
    env = gym.make('BreakoutNoFrameskip-v4')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)

    env.seed(1)
    env.action_space.seed(1)
    env.observation_space.seed(1)

    return env

def test_gym_seed():
    random.seed(1)
    np.random.seed(1)

    #envs = [gym.make('BreakoutNoFrameskip-v4') for _ in range(2)]
    #envs = [make_env() for _ in range(2)]
    binding = pufferlib.registry.Atari('BreakoutNoFrameskip-v4')
    envs = [binding.env_creator() for _ in range(2)]

    for e in envs:
        e.seed(1)

    obs = [e.reset()[1] for e in envs]
    print(np.all(obs[0] == obs[1]))


if __name__ == '__main__':
    test_gym_seed()