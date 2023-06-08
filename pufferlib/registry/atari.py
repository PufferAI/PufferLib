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
        return np.array(obs[1], dtype=np.float32).ravel()

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
            record_episode_statistics=False,
        )

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    '''CleanRL's default layer initialization'''
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(pufferlib.models.Policy):
    def __init__(self, binding, *args, framestack, input_size=512, hidden_size=128, **kwargs):
        '''The CleanRL default Atari policy: a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword arguments. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__(binding)
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
            layer_init(nn.Linear(64 * 7 * 7, input_size)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(hidden_size, self.num_actions), std=0.01)
        self.value_function = layer_init(nn.Linear(hidden_size, 1), std=1)

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
