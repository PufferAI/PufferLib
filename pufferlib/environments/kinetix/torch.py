import torch
from torch import nn

import pufferlib.models

Recurrent = pufferlib.models.LSTMWrapper

from pufferlib.models import Default as Policy
SymbolicPolicy = Policy

class PixelsPolicy(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = 128
        self.map_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(3, cnn_channels, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(cnn_channels, cnn_channels, 4, stride=2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(14 * 14 * cnn_channels, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

        self.is_continuous = False

    def forward(self, observations):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def encode_observations(self, observations):
        encoded = self.map_encoder(observations.permute(0, 3, 1, 2))
        features = self.proj(encoded)
        return features

    def decode_actions(self, flat_hidden, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
