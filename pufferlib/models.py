        
from pdb import set_trace as T
import numpy as np

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces


class Default(nn.Module):
    '''Default PyTorch policy. Flattens obs and applies a linear layer.

    PufferLib is not a framework. It does not enforce a base class.
    You can use any PyTorch policy that returns actions and values.
    We structure our forward methods as encode_observations and decode_actions
    to make it easier to wrap policies with LSTMs. You can do that and use
    our LSTM wrapper or implement your own. To port an existing policy
    for use with our LSTM wrapper, simply put everything from forward() before
    the recurrent cell into encode_observations and put everything after
    into decode_actions.
    '''

    def __init__(self, env, hidden_size=128): # 128
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = np.prod(env.single_observation_space.shape)

        # If emulated, get dtype for potential nativization
        if env.emulated:
            # Nativize the dtype based on the environment's emulated observation space
            self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)

        # Initialize the decoder based on the type of action space
        self.is_multidiscrete = isinstance(env.single_action_space, pufferlib.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        if self.is_multidiscrete:
            action_nvec = env.single_action_space.nvec
            self.decoder = nn.ModuleList([
                pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01) for n in action_nvec
            ])
        elif not self.is_continuous:
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std=0.01
            )
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01
            )
            self.decoder_logstd = nn.Parameter(torch.zeros(1, env.single_action_space.shape[0]))

        # Initialize the value head
        self.value_head = nn.Linear(hidden_size, 1)

        # breakpoint()
        # Delay the initialization of the encoder until we know the input size
        self.encoder = None

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        '''Encodes a batch of observations into hidden states.'''
        
        if isinstance(self.dtype , dict) and len(self.dtype) > 1:
            # Multiple observation spaces, check if dtypes are heterogeneous
            if any(dtype[0] != next(iter(self.dtype.values()))[0] for dtype in self.dtype.values()):
                observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
                concatenated_observations = torch.cat(
                    [v.view(v.shape[0], -1) for v in observations.values()], dim=1
                )
        else:
            # Directly flatten observations if there's only one dtype or uniform dtypes
            concatenated_observations = observations.view(observations.shape[0], -1)

        # Initialize the encoder now that we know the input size
        if self.encoder is None:
            input_size = concatenated_observations.shape[-1]
            self.encoder = nn.Linear(input_size, self.hidden_size)

        # Pass the concatenated observations through the encoder and apply ReLU activation
        hidden = torch.relu(self.encoder(concatenated_observations))
        return hidden, None

    def decode_actions(self, hidden, lookup, concat=True):
        '''Decodes a batch of hidden states into (multi)discrete or continuous actions.
        Assumes no time dimension (handled by LSTM wrappers).'''

        # Compute the value of the hidden state
        value = self.value_head(hidden)

        # Decode actions based on the action space type
        if self.is_multidiscrete:
            actions = [dec(hidden) for dec in self.decoder]
            return actions, value
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            probs = torch.distributions.Normal(mean, std)
            return probs, value

        # For other action spaces, use a single decoder
        actions = self.decoder(hidden)
        return actions, value


        
class LSTMWrapper(nn.Module):
    def __init__(self, env, policy, num_layers=1): # input_size=512, hidden_size=512, 
        '''Wraps your policy with an LSTM without letting you shoot yourself in the
        foot with bad transpose and shape operations. This saves much pain.
        Requires that your policy define encode_observations and decode_actions.
        See the Default policy for an example.'''
        super().__init__()
        self.obs_shape = env.single_observation_space.shape

        self.policy = policy
        
        # TODO: programmatically determine input_size
        self.input_size = 128 # 256 # np.prod(env.single_observation_space.shape)
        self.default = Default(env)
        self.hidden_size = self.default.hidden_size # policy.hidden_size
        self.recurrent = nn.LSTM(self.input_size, self.hidden_size, num_layers)

        for name, param in self.recurrent.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

    def forward(self, x, state):
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        # breakpoint()
        # dummy data:           vs.     real data:
        # x_shape (8,30)                (8,108)
        # space_shape (108,)            (108,)
        
        
        if x_shape[-space_n:] != space_shape:
            raise ValueError('Invalid input tensor shape', x.shape)

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

class Convolutional(nn.Module):
    def __init__(self, env, *args, framestack, flat_size,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1, **kwargs):
        '''The CleanRL default NatureCNN policy used for Atari.
        It's just a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword argument. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample

        self.network= nn.Sequential(
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
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, 1), std=1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

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

class ProcgenResnet(nn.Module):
    '''Procgen baseline from the AICrowd NeurIPS 2020 competition
    Based on the ResNet architecture that was used in the Impala paper.'''
    def __init__(self, env, cnn_width=16, mlp_width=256):
        super().__init__()
        h, w, c = env.single_observation_space.shape
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
                nn.Linear(mlp_width, env.single_action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
                nn.Linear(mlp_width, 1), std=1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        return hidden, None
 
    def decode_actions(self, hidden, lookup):
        '''linear decoder function'''
        action = self.actor(hidden)
        value = self.value(hidden)
        return action, value

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
