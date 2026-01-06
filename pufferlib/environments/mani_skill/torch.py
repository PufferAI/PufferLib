import numpy as np

import torch
import torch.nn as nn

import pufferlib
from pufferlib.models import Default as Policy
from pufferlib.models import LSTMWrapper as Recurrent

class FakePolicy(nn.Module):
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
    def __init__(self, env, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size

        n_obs = np.prod(env.single_observation_space.shape)
        n_atn = env.single_action_space.shape[0]
        self.decoder_mean = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(n_obs, 256)),
            nn.Tanh(),
            pufferlib.pytorch.layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            pufferlib.pytorch.layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            pufferlib.pytorch.layer_init(nn.Linear(256, n_atn), std=0.01),
        )
        self.decoder_logstd = nn.Parameter(torch.zeros(
            1, env.single_action_space.shape[0]))

        self.value = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(n_obs, 256)),
            nn.Tanh(),
            pufferlib.pytorch.layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            pufferlib.pytorch.layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            pufferlib.pytorch.layer_init(nn.Linear(256, 1), std=1),
        )
 
    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        '''Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers).'''
        return observations

    def decode_actions(self, hidden):
        '''Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers).'''
        mean = self.decoder_mean(hidden)
        logstd = self.decoder_logstd.expand_as(mean)
        std = torch.exp(logstd)
        logits = torch.distributions.Normal(mean, std)
        values = self.value(hidden)
        return logits, values
