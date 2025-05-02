import torch
from torch import nn

import pufferlib.models

Recurrent = pufferlib.models.LSTMWrapper

'''
CRAFTAX_CHANNELS = 83

# Are these transposed?
CRAFTAX_ROWS = 11
CRAFTAX_COLS = 9

N_MAP = CRAFTAX_ROWS * CRAFTAX_COLS * CRAFTAX_CHANNELS
N_FLAT = 51
'''

CRAFTAX_ROWS = 7
CRAFTAX_COLS = 9
CRAFTAX_CHANNELS = 21
N_MAP = CRAFTAX_ROWS * CRAFTAX_COLS * CRAFTAX_CHANNELS
N_FLAT = 22


class Policy(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.map_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(21, cnn_channels, 3, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.flat_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(N_FLAT, hidden_size)),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(2*cnn_channels + hidden_size, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

        self.is_continuous = False

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        map_obs = observations[:, :N_MAP].view(
            -1, CRAFTAX_ROWS, CRAFTAX_COLS, CRAFTAX_CHANNELS
            ).permute(0, 3, 1, 2)
        map_obs = self.map_encoder(map_obs)
        flat_obs = observations[:, N_MAP:]
        flat_obs = self.flat_encoder(flat_obs)
        features = torch.cat([map_obs, flat_obs], dim=1)
        features = self.proj(features)
        return features, None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
