from pdb import set_trace as T

import torch
import torch.nn as nn
import torch.nn.functional as F

import pufferlib.models
import pufferlib.pytorch
from pufferlib.pytorch import layer_init


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=256, hidden_size=256, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

class Policy(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)

        self.blstats_net = nn.Sequential(
            nn.Embedding(256, 32),
            nn.Flatten(),
        )

        self.char_embed = nn.Embedding(256, 32)
        self.chars_net = nn.Sequential(
            layer_init(nn.Conv2d(32, 32, 5, stride=(2, 3))),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 5, stride=(1, 3))),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.proj = nn.Linear(864+960, 256)
        self.actor = layer_init(nn.Linear(256, 8), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        hidden = self.encode_observations(x)
        actions, value = self.decode_actions(hidden, None)
        return actions, value

    def encode_observations(self, x):
        x = x.type(torch.uint8) # Undo bad cleanrl cast
        x = pufferlib.pytorch.nativize_tensor(x, self.dtype)

        blstats = torch.clip(x['blstats'] + 1, 0, 255).int()
        blstats = self.blstats_net(blstats)

        chars = self.char_embed(x['chars'].int())
        chars = torch.permute(chars, (0, 3, 1, 2))
        chars = self.chars_net(chars)

        concat = torch.cat([blstats, chars], dim=1)
        return self.proj(concat)

    def decode_actions(self, hidden, lookup, concat=None):
        value = self.critic(hidden)
        action = self.actor(hidden)
        return action, value
