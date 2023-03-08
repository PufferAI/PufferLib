from pdb import set_trace as T

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Policy(torch.nn.Module, ABC):
    def __init__(self, binding, input_size, hidden_size):
        '''Pure PyTorch base policy
        
        This spec allows PufferLib to repackage your policy
        for compatibility with RL frameworks
        '''
        super().__init__()
        self.binding = binding
        self.input_size = input_size
        self.hidden_size = hidden_size

    @abstractmethod
    def critic(self, hidden):
        '''Computes the value function from the hidden state
        
        Returns a single value for each batch element'''
        raise NotImplementedError

    @abstractmethod
    def encode_observations(self, flat_observations):
        '''Encodes observations into a flat vector.
        
        Call pufferlib.emulation.unpack_batched_obs to unflatten obs
        Returns a tuple of (hidden, lookup) where lookup is an optional
        mechanism to return additional encoding information
        '''
        raise NotImplementedError

    @abstractmethod
    def decode_actions(self, flat_hidden, lookup):
        '''Decodes hidden state into a multidiscrete action space
        
        Returns a tensor of logits for each action space dimension'''
        raise NotImplementedError

class Default(Policy):
    def __init__(self, binding, input_size=256, hidden_size=256):
        '''Default PyTorch policy
        
        It is just a linear layer over the flattened obs with
        linear action decoders. This is for debugging only.
        It is not a good policy for almost any env.
        '''
        super().__init__(binding, input_size, hidden_size)
        self.encoder = nn.Linear(self.binding.single_observation_space.shape[0], hidden_size)
        self.decoders = nn.ModuleList([nn.Linear(hidden_size, n)
                for n in self.binding.single_action_space.nvec])
        self.value_head = nn.Linear(hidden_size, 1)

    def critic(self, hidden):
        return self.value_head(hidden)

    def encode_observations(self, env_outputs):
        return self.encoder(env_outputs), None

    def decode_actions(self, hidden, lookup, concat=True):
        actions = [dec(hidden) for dec in self.decoders]
        if concat:
            return torch.cat(actions, dim=-1)
        return actions