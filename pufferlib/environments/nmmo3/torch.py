from pdb import set_trace as T
import torch
from torch import nn
import numpy as np

import pufferlib.models
import pufferlib.pytorch

class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=256, hidden_size=256, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

@torch.compiler.disable
def decode_map(codes):
    codes = codes.unsqueeze(1).long()
    factors = [4, 4, 16, 5, 3, 5, 5, 6, 7, 4]
    n_channels = sum(factors)
    obs = torch.zeros(codes.shape[0], n_channels, 11, 15, device='cuda')

    add, div = 0, 1
    # TODO: check item/tier order
    for mod in factors:
        obs.scatter_(1, add+(codes//div)%mod, 1)
        add += mod
        div *= mod

    return obs

class Decompressor(nn.Module):
    def __init__(self):
        super().__init__()
        factors = np.array([4, 4, 16, 5, 3, 5, 5, 6, 7, 4])
        n_channels = sum(factors)
        add = np.array([0, *np.cumsum(factors).tolist()[:-1]])[None, :, None]
        div = np.array([1, *np.cumprod(factors).tolist()[:-1]])[None, :, None]

        factors = torch.tensor(factors)[None, :, None].cuda()
        add = torch.tensor(add).cuda()
        div = torch.tensor(div).cuda()

        self.register_buffer('factors', factors)
        self.register_buffer('add', add)
        self.register_buffer('div', div)

    def forward(self, codes):
        batch = codes.shape[0]
        obs = torch.zeros(batch, 59, 11*15, device=codes.device)
        codes = codes.view(codes.shape[0], 1, -1)
        dec = self.add + (codes//self.div) % self.factors
        obs.scatter_(1, dec, 1)
        return obs.view(batch, 59, 11, 15)


class Policy(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.emulated = env.emulated
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        #self.env = env

        self.num_actions = env.single_action_space.n
        #self.num_actions = self.action_space.n
        #self.num_actions = self.action_space.shape[0]
        hidden_size = 256
        output_size = 256

        self.decompressor = Decompressor()

        self.map_2d = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(59, 64, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(128, hidden_size//2)),
            nn.ReLU(),
        )

        self.embed = nn.Embedding(128, 32)
        self.player_1d = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(32*44, hidden_size//2)),
            nn.ReLU(),
        )

        self.proj = nn.Linear(hidden_size, output_size)

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(output_size, self.num_actions), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

    def forward(self, x):
        hidden, lookup = self.encode_observations(x)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations, unflatten=False):
        batch = observations.shape[0]

        #x = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
        #with torch.no_grad():
        #    ob_map = self.decompressor(x['map']).float()
        #player = x['player']

        ob_map = observations[:, :(11*15)].view(batch, 11, 15)
        ob_map = self.decompressor(ob_map).float()
        player = observations[:, (11*15):]

        ob_map = self.map_2d(ob_map)
        ob_player = self.embed(player)
        ob_player = ob_player.flatten(1)
        ob_player = self.player_1d(ob_player)

        ob = torch.cat([ob_map, ob_player], dim=1)
        return self.proj(ob), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        #action = [self.actor(flat_hidden)]
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
