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
import pufferlib.binding


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

def env_creator(env_name, framestack):
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

def create_binding(env_name, framestack):
    return pufferlib.emulation.Binding(
        env_creator=env_creator,
        default_args=[env_name, framestack],
        env_name=env_name,
        env_includes_reset=True,
        emulate_flat_atn=True,
    )

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(pufferlib.binding.Policy):
    def __init__(self, binding, *args, framestack, input_size=512, hidden_size=512, **kwargs):
        super().__init__(input_size, hidden_size, *args, **kwargs)
        self.observation_space = binding.raw_single_observation_space
        self.num_actions = binding.raw_single_action_space.n

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, self.hidden_size)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(self.hidden_size, self.num_actions), std=0.01)
        self.value_function = layer_init(nn.Linear(self.hidden_size, 1), std=1)

    def critic(self, hidden):
        return self.value_function(hidden)

    def encode_observations(self, flat_observations):
        # TODO: Add flat obs support to emulation
        batch = flat_observations.shape[0]
        observations = flat_observations.reshape((batch,) + self.observation_space.shape)
        return self.network(observations / 255.0), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        if concat:
            return action
        return [action]

