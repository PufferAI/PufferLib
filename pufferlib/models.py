from pdb import set_trace as T

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.distributions import Categorical

import pufferlib.pytorch


#                 recurrent_cls=pufferlib.pytorch.BatchFirstLSTM,
#                 recurrent_kwargs=dict(input_size=128, hidden_size=256)
 
class Policy(torch.nn.Module, ABC):
    def __init__(self, binding, recurrent_cls=None, recurrent_kwargs={}):
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

        Args:
            binding: A pufferlib.emulation.Binding object
        '''
        super().__init__()
        self.binding = binding
        self.recurrent_cls = recurrent_cls
        if recurrent_cls is not None:
            self.recurrent_policy = recurrent_cls(**recurrent_kwargs)

            # TODO: Generalize this
            self.input_size = recurrent_kwargs['input_size']
            self.hidden_size = recurrent_kwargs['hidden_size']
            for name, param in self.recurrent_policy.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)
 
    @abstractmethod
    def critic(self, hidden):
        '''Computes the value function from the hidden state
        
        Returns a single value for each batch element'''
        raise NotImplementedError

    @abstractmethod
    def encode_observations(self, flat_observations):
        '''Encodes a batch of observations into hidden states

        Call pufferlib.emulation.unpack_batched_obs at the start of this
        function to unflatten observations to their original structured form:

        observations = pufferlib.emulation.unpack_batched_obs(
            self.binding.raw_single_observation_space, env_outputs)
 
        Args:
            flat_observations: A tensor of shape (batch, ..., obs_size)

        Returns:
            hidden: Tensor of (batch, ..., hidden_size)
            lookup: Tensor of (batch, ...) that can be used to return additional embeddings
        '''
        raise NotImplementedError

    def forward(self, x, state):
        if state is not None:
            assert len(state) == 2

        assert len(x.shape) == 3
        B, TT, _ = x.shape

        x = x.reshape(B*TT, -1)

        hidden, lookup = self.encode_observations(x)
        assert hidden.shape == (B*TT, self.input_size)

        hidden = hidden.view(B, TT, self.input_size)
        hidden, state = self.recurrent_policy(hidden, state)
        hidden = hidden.reshape(B*TT, self.hidden_size)

        return hidden, state, lookup
 
    @abstractmethod
    def decode_actions(self, flat_hidden, lookup):
        '''Decodes a batch of hidden states into multidiscrete actions

        Args:
            flat_hidden: Tensor of (batch, ..., hidden_size)
            lookup: Tensor of (batch, ...), if returned by encode_observations

        Returns:
            actions: Tensor of (batch, ..., action_size)

        actions is a concatenated tensor of logits for each action space dimension.
        It should be of shape (batch, ..., sum(action_space.nvec))'''
        raise NotImplementedError

    @property
    def lstm(self):
        return self.recurrent_policy

    def get_value(self, x, state=None, done=None):
        x = x.reshape((-1, x.shape[0], x.shape[-1]))
        hidden, state, lookup = self.forward(x, state)
        return self.critic(hidden)

    # TODO: Compute seq_lens from done
    def get_action_and_value(self, x, state=None, done=None, action=None):
        x = x.reshape((-1, x.shape[0], x.shape[-1]))
        hidden, state, lookup = self.forward(x, state)
        value = self.critic(hidden)
        flat_logits = self.decode_actions(hidden, lookup, concat=False)
        multi_categorical = [Categorical(logits=l) for l in flat_logits]

        if action is None:
            action = torch.stack([c.sample() for c in multi_categorical])
        else:
            action = action.view(-1, action.shape[-1]).T

        logprob = torch.stack([c.log_prob(a) for c, a in zip(multi_categorical, action)]).T.sum(1)
        entropy = torch.stack([c.entropy() for c in multi_categorical]).T.sum(1)

        return action.T, logprob, entropy, value, state

class Default(Policy):
    def __init__(self, binding, input_size=128, hidden_size=128):
        '''Default PyTorch policy, meant for debugging.
        This should run with any environment but is unlikely to learn anything.
        
        Uses a single linear layer to encode observations and a list of
        linear layers to decode actions. The value function is a single linear layer.
        '''
        super().__init__(binding)
        self.encoder = nn.Linear(self.binding.single_observation_space.shape[0], hidden_size)
        self.decoders = nn.ModuleList([nn.Linear(hidden_size, n)
                for n in self.binding.single_action_space.nvec])
        self.value_head = nn.Linear(hidden_size, 1)

    def critic(self, hidden):
        '''Linear value function'''
        return self.value_head(hidden)

    def encode_observations(self, env_outputs):
        '''Linear encoder function'''
        return self.encoder(env_outputs), None

    def decode_actions(self, hidden, lookup, concat=True):
        '''Concatenated linear decoder function'''
        actions = [dec(hidden) for dec in self.decoders]
        if concat:
            return torch.cat(actions, dim=-1)
        return actions

class Convolutional(Policy):
    def __init__(self, binding, *args, framestack, flat_size, input_size=512, hidden_size=512, output_size=512, **kwargs):
        '''The CleanRL default Atari policy: a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword arguments. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__(binding)
        self.observation_space = binding.single_observation_space
        self.num_actions = binding.raw_single_action_space.n

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
        self.value_function = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

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