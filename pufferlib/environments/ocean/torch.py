from torch import nn
import torch.nn.functional as F

from functools import partial
import pufferlib.models

Recurrent = pufferlib.models.LSTMWrapper
#Policy = pufferlib.models.Default

class Policy(nn.Module):
    #def __init__(self, env, flat_size=144,
    def __init__(self, env, flat_size=32,
            input_size=32, hidden_size=128, output_size=128,
            downsample=1, **kwargs):
        super().__init__()
        '''
        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(5, 32, 8, stride=4, padding=(1, 1))),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 16, 3, stride=2)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
        )
        '''
        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(8, 32, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 32, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, 1), std=1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        observations = F.one_hot(observations.long(), 8).permute(0, 3, 1, 2).float()
        return self.network(observations), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

