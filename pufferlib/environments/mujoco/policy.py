import numpy as np
import torch
import torch.nn as nn

import pufferlib
from pufferlib.pytorch import layer_init

from pufferlib.models import Default as Policy


# Puffer LSTMWrapper does NOT support separate critic networks for now
# Would be good to test he performance between these architectures
class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=128, hidden_size=128, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)


# Difference between CleanRL and PufferRL policies is that
# CleanRL has seperate actor and critic networks, while
# PufferRL has a single encoding network that is shared between
# the actor and critic networks
class CleanRLPolicy(torch.nn.Module):
    def __init__(self, env, hidden_size=64):
        super().__init__()
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        self.actor_encoder = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.actor_decoder_mean = layer_init(
            nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01
        )
        self.actor_decoder_logstd = nn.Parameter(torch.zeros(1, env.single_action_space.shape[0]))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def forward(self, observations):
        observations = observations.float()
        hidden, lookup = self.encode_observations(observations)
        actions, _ = self.decode_actions(hidden, lookup)
        value = self.critic(observations)
        return actions, value

    # NOTE: these are for the LSTM wrapper, which may NOT work as intended
    def encode_observations(self, observations):
        """Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers)."""
        batch_size = observations.shape[0]
        observations = observations.view(batch_size, -1)
        return self.actor_encoder(observations), None

    def decode_actions(self, hidden, lookup, concat=True):
        """Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers)."""
        #value = self.value_head(hidden)

        mean = self.actor_decoder_mean(hidden)
        logstd = self.actor_decoder_logstd.expand_as(mean)
        std = torch.exp(logstd)
        probs = torch.distributions.Normal(mean, std)
        # batch = hidden.shape[0]

        return probs, 0  # NOTE: value comes form the separate critic network
