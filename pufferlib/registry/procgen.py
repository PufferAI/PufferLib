from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.distributions.categorical import Categorical

import gym

import pufferlib
from pufferlib.pytorch import layer_init
import pufferlib.emulation
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
    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
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
        self.actor = layer_init(nn.Linear(256, self.action_space.n), std=0.01)
        self.value = layer_init(nn.Linear(256, 1), std=1)

    def encode_observations(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        return hidden, None
 
    def decode_actions(self, hidden, lookup):
        '''linear decoder function'''
        action = self.actor(hidden)
        value = self.value(hidden)
        return action, value


class ProcgenVecEnv:
    '''WIP Vectorized Procgen environment wrapper
    
    Does not use normal PufferLib emulation'''
    def __init__(self, env_name, num_envs=1,
            num_levels=0, start_level=0,
            distribution_mode="easy"):

        self.num_envs = num_envs
        self.envs = ProcgenEnv(
            env_name=env_name,
            num_envs=num_envs,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
        )

    @property
    def single_observation_space(self):
        return self.envs.observation_space['rgb']

    @property
    def single_action_space(self):
        return self.envs.action_space

    def reset(self, seed=None):
        obs = self.envs.reset()['rgb']
        rewards = [0] * self.num_envs
        dones = [False] * self.num_envs
        infos = [{}] * self.num_envs
        return obs, rewards, dones, infos

    def step(self, actions):
        actions = np.array(actions)
        obs, rewards, dones, infos = self.envs.step(actions)
        return obs['rgb'], rewards, dones, infos


class ProcgenPostprocessor(pufferlib.emulation.Postprocessor):
    def features(self, obs):
        try:
            return obs['rgb']
        except:
            return obs

    def reward_done_info(self, reward, done, info):
        return float(reward), bool(done), info


def make_env(name):
    '''Atari creation function with default CleanRL preprocessing based on Stable Baselines3 wrappers'''
    try:
        with pufferlib.utils.Suppress():
            import gym3
            from procgen.env import ProcgenGym3Env
            env = ProcgenGym3Env(num=1, env_name=name)
    except ImportError as e:
        raise e('Cannot gym.make ALE environment (pip install pufferlib[gym])')

    env = gym3.ToGymEnv(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: obs["rgb"])
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.NormalizeReward(env, gamma=0.999)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    env = pufferlib.emulation.GymPufferEnv(
        env=env,
        postprocessor_cls=ProcgenPostprocessor,
    )
    return env
