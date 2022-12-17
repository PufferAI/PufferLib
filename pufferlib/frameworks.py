from pdb import set_trace as T

from abc import ABC, abstractmethod

import torch

from pufferlib.torch import BatchFirstLSTM


class BasePolicy(torch.nn.Module, ABC):
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