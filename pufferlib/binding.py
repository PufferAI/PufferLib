from pdb import set_trace as T
import numpy as np

from abc import ABC, abstractmethod

import inspect

import torch
import torch.nn as nn

from pufferlib.emulation import PufferWrapper
from pufferlib.torch import BatchFirstLSTM


def auto(env_name=None, env=None, env_cls=None, env_args=[], env_kwargs={}, **kwargs):
    '''Create binding for the specified environment
    
    Args:
        env_name: Name of the environment. Inferred from the class if not specified.
        env: Environment object to wrap. Alternatively, specify env_cls
        env_cls: Environment class to wrap. Alternatively, specify env
        env_args: Arguments for env_cls
        env_kwargs: Keyword arguments for env_cls
        **kwargs: Allows you to specify PufferWrapper arguments directly

    Returns:
        A PufferLib binding with the following properties:
            env_creator: Function that creates a wrapped environment
            single_observation_space: Observation space of a single agent
            single_action_space: Action space of a single agent
    '''
    class FlatNetwork(Policy):
        def __init__(self, observation_space, action_space,
                input_size, hidden_size, lstm_layers):
            '''Default PyTorch policy
            
            It's just a linear layer over the flattened obs with
            linear action decoders. This is for debugging only.
            It is not a good policy for almost any env.
            
            Instantiated for you by the auto wrapper
            '''
            super().__init__(input_size, hidden_size, lstm_layers)
            self.observation_space = observation_space
            self.action_space = action_space
            self.encoder = nn.Linear(observation_space.shape[0], hidden_size)
            self.decoders = nn.ModuleList([nn.Linear(hidden_size, n)
                    for n in action_space.nvec])
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

    class AutoBound(Base):
        def __init__(self):
            if env_cls is not None:
                #assert inspect.isclass(env_cls)
                self.env_cls = PufferWrapper(env_cls, **kwargs)
                self.env_name = env_cls.__name__
            elif env is not None:
                assert not inspect.isclass(env)
                self.env_cls = PufferWrapper(env, **kwargs)
                self.env_name = env.__class__.__name__
            else:
                raise Exception('Specify env or env_cls')

            if env_name is not None:
                self.env_name = env_name

            super().__init__(self.env_name, self.env_cls, env_args, env_kwargs)
            self.policy = FlatNetwork

    return AutoBound()

class Base:
    def __init__(self, env_name, env_cls, env_args=[], env_kwargs={}):
        '''Base class for PufferLib bindings

        Args: 
            env_name: Name of the environment
            env_cls: Environment class to wrap
            env_args: Arguments for env_cls
            env_kwargs: Keyword arguments for env_cls
        '''
        self.env_name = env_name
        self.env_cls = env_cls

        self.env_args = env_args
        self.env_kwargs = env_kwargs

        self.env_creator = lambda: self.env_cls(*self.env_args, **self.env_kwargs)

        local_env = self.env_creator()
        self.default_agent = local_env.possible_agents[0]
        self.single_observation_space = local_env.observation_space(self.default_agent)
        self.single_action_space = local_env.action_space(self.default_agent)

    @property
    def custom_model_config(self):
        '''Custom Policy arguments for this environment'''
        return {
            'observation_space': self.single_observation_space,
            'action_space': self.single_action_space,
            'input_size': 32,
            'hidden_size': 32,
            'lstm_layers': 0,
        }


class Policy(torch.nn.Module, ABC):
    def __init__(self, input_size, hidden_size, lstm_layers=0):
        '''Pure PyTorch base policy
        
        This spec allows PufferLib to repackage your policy
        for compatibility with RL frameworks
        '''
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

    @abstractmethod
    def critic(self, hidden):
        '''Computes the value function from the hidden state'''
        raise NotImplementedError
        return hidden

    @abstractmethod
    def encode_observations(self, flat_observations):
        '''Encodes observations into a flat vector.
        
        Call pufferlib.emulation.unpack_batched_obs to unflatten obs
        '''
        raise NotImplementedError
        return hidden, lookup

    @abstractmethod
    def decode_actions(self, flat_hidden, lookup):
        '''Decodes hidden state into a multidiscrete action space'''
        raise NotImplementedError
        return logits

def make_recurrent_policy(Policy, batch_first=True):
    '''Wraps the given policy with an LSTM
    
    Called for you by framework-specific bindings
    '''
    class Recurrent(Policy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert self.lstm_layers > 0
            if batch_first:
                self.lstm = BatchFirstLSTM(
                    self.input_size,
                    self.hidden_size,
                    self.lstm_layers,
                )
            else:
                self.lstm = torch.nn.LSTM(
                    self.input_size,
                    self.hidden_size,
                    self.lstm_layers,
                )
 
        def encode_observations(self, x, state):
            # TODO: Check shapes
            assert state is not None

            assert len(state) == 2
            assert len(x.shape) == 3

            B, TT, _ = x.shape
            x = x.reshape(B*TT, -1)

            hidden, lookup = super().encode_observations(x)
            assert hidden.shape == (B*TT, self.hidden_size)

            hidden = hidden.view(B, TT, self.hidden_size)
            hidden, state = self.lstm(hidden, state)
            hidden = hidden.reshape(B*TT, self.hidden_size)

            return hidden, state, lookup
    return Recurrent

