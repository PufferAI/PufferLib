from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

import pufferlib.emulation
from pufferlib.models import Policy as Base

class Policy(Base):
    def __init__(self, env, input_size=128, hidden_size=128):
        '''Default PyTorch policy, meant for debugging.
        This should run with any environment but is unlikely to learn anything.
        
        Uses a single linear layer + relu to encode observations and a list of
        linear layers to decode actions. The value function is a single linear layer.
        '''
        super().__init__(env)

        self.flat_observation_space = env.flat_observation_space
        self.flat_observation_structure = env.flat_observation_structure

        self.encoder = nn.Linear(np.prod(
            env.structured_observation_space['obs'].shape), hidden_size)
        self.decoder = nn.Linear(hidden_size, self.action_space.n)

        self.value_head = nn.Linear(hidden_size, 1)

    def encode_observations(self, observations):
        '''Linear encoder function'''
        observations = pufferlib.emulation.unpack_batched_obs(observations,
            self.flat_observation_space, self.flat_observation_structure)
        obs = observations['obs'].view(observations['obs'].shape[0], -1)
        self.action_mask = observations['action_mask']

        hidden = torch.relu(self.encoder(obs))
        return hidden, None

    def decode_actions(self, hidden, lookup, concat=True):
        '''Concatenated linear decoder function'''
        value = self.value_head(hidden)
        action = self.decoder(hidden)
        action = action.masked_fill(self.action_mask == 0, -1e9)
        return action, value