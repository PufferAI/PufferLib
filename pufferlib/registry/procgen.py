from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.distributions.categorical import Categorical

import gym
from procgen import ProcgenEnv

import pufferlib
from pufferlib.pytorch import layer_init
import pufferlib.emulation
import pufferlib.models


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class Policy(pufferlib.models.Policy):
    def __init__(self, binding, input_size, output_size):
        super().__init__(binding)
        h, w, c = binding.raw_single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init(nn.Linear(256, binding.raw_single_action_space.n), std=0.01)
        self.value = layer_init(nn.Linear(256, 1), std=1)

    def critic(self, x):
        # TODO: Separate critic and actor
        return self.value(x)

    def encode_observations(self, x):
        x = self.binding.unpack_batched_obs(x)[0].squeeze(1)
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        return hidden, None
 
    def decode_actions(self, hidden, lookup, concat=True):
        '''linear decoder function'''
        actions = self.actor(hidden)
        if concat:
            return actions
        return [actions]


class PufferWrapper(gym.Env):
    def __init__(self, envs):
        self.envs = envs
        self.observation_space = envs.single_observation_space
        self.action_space = envs.single_action_space
    
    def step(self, actions):
        obs, rewards, dones, infos = self.envs.step(np.array([actions]))
        return obs, rewards[0], dones[0], infos[0]

    def reset(self):
        return self.envs.reset()
    
    def close(self):
        return self.envs.close()


def make_env(env_name):
    '''Atari creation function with default CleanRL preprocessing based on Stable Baselines3 wrappers'''
    try:
        with pufferlib.utils.Suppress():
            envs = ProcgenEnv(num_envs=1, env_name=env_name, num_levels=0, start_level=0, distribution_mode="easy")
    except ImportError as e:
        raise e('Cannot gym.make ALE environment (pip install pufferlib[gym])')

    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.observation_space = envs.observation_space["rgb"]
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs, gamma=0.999)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    envs = PufferWrapper(envs)
    return envs


def make_binding(env_name):
    '''Procgen binding creation function'''
    try:
        make_env(env_name)
    except:
        raise pufferlib.utils.SetupError(f'{env_name} (procgen)')
    else:
        return pufferlib.emulation.Binding(
            env_creator=make_env,
            default_args=[env_name],
            env_name=env_name,
            emulate_flat_atn=True,
            suppress_env_prints=False,
        )