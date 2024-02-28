from pdb import set_trace as T
import numpy as np

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces


class Policy(nn.Module):
    '''Pure PyTorch base policy
    
    This spec allows PufferLib to repackage your policy
    for compatibility with RL frameworks

    encode_observations -> decode_actions is PufferLib's equivalent of PyTorch forward
    This structure provides additional flexibility for us to include an LSTM
    between the encoder and decoder.

    To port a policy to PufferLib, simply put everything from forward() before the
    recurrent cell (or, if no recurrent cell, everything before the action head)
    into encode_observations and put everything after into decode_actions.

    You can delete the recurrent cell from forward(). PufferLib handles this for you
    with its framework-specific wrappers. Since each frameworks treats temporal data a bit
    differently, this approach lets you write a single PyTorch network for multiple frameworks.

    Specify the value function in critic(). This is a single value for each batch element.
    It is called on the output of the recurrent cell (or, if no recurrent cell,
    the output of encode_observations)
    '''
    def __init__(self, env):
        super().__init__()
        if isinstance(env, pufferlib.emulation.GymnasiumPufferEnv):
            self.observation_space = env.observation_space
            self.action_space = env.action_space
        else:
            self.observation_space = env.single_observation_space
            self.action_space = env.single_action_space

        # Used to unflatten observation in forward pass
        self.unflatten_context = env.unflatten_context

        self.is_multidiscrete = isinstance(self.action_space,
                pufferlib.spaces.MultiDiscrete)

        if not self.is_multidiscrete:
            assert isinstance(self.action_space, pufferlib.spaces.Discrete)

    @abstractmethod
    def encode_observations(self, flat_observations):
        '''Encodes a batch of observations into hidden states

        Call pufferlib.emulation.unpack_batched_obs at the start of this
        function to unflatten observations to their original structured form:

        observations = pufferlib.emulation.unpack_batched_obs(
            env_outputs, self.unflatten_context)
 
        Args:
            flat_observations: A tensor of shape (batch, ..., obs_size)

        Returns:
            hidden: Tensor of (batch, ..., hidden_size)
            lookup: Tensor of (batch, ...) that can be used to return additional embeddings
        '''
        raise NotImplementedError

    @abstractmethod
    def decode_actions(self, flat_hidden, lookup):
        '''Decodes a batch of hidden states into multidiscrete actions

        Args:
            flat_hidden: Tensor of (batch, ..., hidden_size)
            lookup: Tensor of (batch, ...), if returned by encode_observations

        Returns:
            actions: Tensor of (batch, ..., action_size)
            value: Tensor of (batch, ...)

        actions is a concatenated tensor of logits for each action space dimension.
        It should be of shape (batch, ..., sum(action_space.nvec))'''
        raise NotImplementedError

    def forward(self, env_outputs):
        '''Forward pass for PufferLib compatibility'''
        hidden, lookup = self.encode_observations(env_outputs)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

class RecurrentWrapper(Policy):
    def __init__(self, env, policy, input_size=128, hidden_size=128, num_layers=1):
        super().__init__(env)

        if not isinstance(policy, Policy):
            raise ValueError('Subclass pufferlib.Policy to use RecurrentWrapper')

        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.recurrent = torch.nn.LSTM(
            input_size, hidden_size, num_layers=num_layers)

        for name, param in self.recurrent.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

    def forward(self, x, state):
        x_shape, space_shape = x.shape, self.observation_space.shape
        x_n, space_n = len(x_shape), len(space_shape)
        assert x_shape[-space_n:] == space_shape

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError('Invalid input tensor shape', x.shape)

        if state is not None:
            assert state[0].shape[1] == state[1].shape[1] == B

        x = x.reshape(B*TT, *space_shape)
        hidden, lookup = self.policy.encode_observations(x)
        assert hidden.shape == (B*TT, self.input_size)

        hidden = hidden.reshape(B, TT, self.input_size)
        hidden = hidden.transpose(0, 1)
        hidden, state = self.recurrent(hidden, state)

        hidden = hidden.transpose(0, 1)
        hidden = hidden.reshape(B*TT, self.hidden_size)

        hidden, critic = self.policy.decode_actions(hidden, lookup)
        return hidden, critic, state

class Default(Policy):
    def __init__(self, env, input_size=128, hidden_size=128):
        '''Default PyTorch policy, meant for debugging.
        This should run with any environment but is unlikely to learn anything.
        
        Uses a single linear layer + relu to encode observations and a list of
        linear layers to decode actions. The value function is a single linear layer.
        '''
        super().__init__(env)
        self.encoder = nn.Linear(np.prod(self.observation_space.shape), hidden_size)

        if self.is_multidiscrete:
            self.decoders = nn.ModuleList([nn.Linear(hidden_size, n)
                for n in self.action_space.nvec])
        else:
            self.decoder = nn.Linear(hidden_size, self.action_space.n)

        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, env_outputs):
        '''Forward pass for PufferLib compatibility'''
        hidden, lookup = self.encode_observations(env_outputs)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        '''Linear encoder function'''
        hidden = observations.reshape(observations.shape[0], -1).float()
        hidden = torch.relu(self.encoder(hidden))
        return hidden, None

    def decode_actions(self, hidden, lookup, concat=True):
        '''Concatenated linear decoder function'''
        value = self.value_head(hidden)

        if self.is_multidiscrete:
            actions = [dec(hidden) for dec in self.decoders]
            return actions, value

        action = self.decoder(hidden)
        return action, value

class Convolutional(Policy):
    def __init__(self, env, *args, framestack, flat_size,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1, **kwargs):
        '''The CleanRL default Atari policy: a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword arguments. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__(env)
        self.num_actions = self.action_space.n
        self.channels_last = channels_last
        self.downsample = downsample

        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
        )

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(output_size, self.num_actions), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

    def encode_observations(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.network(observations.float() / 255.0), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

# ResNet Procgen baseline 
# https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
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

class ProcgenResnet(Policy):
    def __init__(self, env, cnn_width=16, mlp_width=256):
        super().__init__(env)
        h, w, c = env.structured_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [cnn_width, 2*cnn_width, 2*cnn_width]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=mlp_width),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(mlp_width, self.action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
                nn.Linear(mlp_width, 1), std=1)

    def encode_observations(self, x):
        x = pufferlib.emulation.unpack_batched_obs(x, self.unflatten_context)
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        return hidden, None
 
    def decode_actions(self, hidden, lookup):
        '''linear decoder function'''
        action = self.actor(hidden)
        value = self.value(hidden)
        return action, value