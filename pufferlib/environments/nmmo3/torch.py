from pdb import set_trace as T
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import pufferlib.models
import pufferlib.pytorch


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=384, hidden_size=256, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

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

class PlayerProjEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.player_embed = nn.Embedding(128, 32)
        self.player_continuous = pufferlib.pytorch.layer_init(
            nn.Linear(47, hidden_size//2))
        self.discrete_proj = pufferlib.pytorch.layer_init(
            nn.Linear(32*47, hidden_size//2))
        self.player_proj = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, hidden_size//2))
 
    def forward(self, player):
        # TODO: slow
        player = player.float()
        player[:, -2:] /= 10
        player = player.int()
        player_discrete = self.player_embed(player).view(player.shape[0], -1)
        player_discrete = self.discrete_proj(player_discrete)
        player_continuous = self.player_continuous(player.float() / 99)
        player = torch.cat([player_discrete, player_continuous], dim=1)
        player = F.relu(player)
        player = self.player_proj(player)
        player = F.relu(player)
        return player


class PlayerEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.player_embed = nn.Embedding(128, hidden_size//4)
        self.player_continuous = pufferlib.pytorch.layer_init(
            nn.Linear(47, hidden_size//4))
        self.player_proj = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, hidden_size//4))
 
    def forward(self, player):
        player[-2:] /= 10
        player_discrete = self.player_embed(player).max(dim=1)[0]
        player_continuous = self.player_continuous(player.float() / 99)
        player = torch.cat([player_discrete, player_continuous], dim=1)
        #player = F.relu(player)
        #player = self.player_proj(player)
        #player = F.relu(player)
        return player

class Policy(nn.Module):
    def __init__(self, env, hidden_size=256, output_size=256):
        super().__init__()
        #self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.num_actions = env.single_action_space.n
        self.decompressor = Decompressor()
        self.factors = np.array([4, 4, 17, 5, 3, 5, 5, 5, 7, 4])

        self.map_2d = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(59, 64, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.Flatten(),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(128, hidden_size//2)),
            nn.ReLU(),
        )

        self.player_encoder = PlayerProjEncoder(hidden_size)
        #self.proj = nn.Linear(hidden_size, output_size)
        #self.player_proj = nn.Linear(47, hidden_size)

        self.reward_proj = nn.Linear(10, hidden_size//2)

        #self.lstm = nn.LSTMCell(hidden_size, hidden_size)

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, self.num_actions), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

    def forward(self, x):
        hidden, lookup = self.encode_observations(x)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations, unflatten=False):
        batch = observations.shape[0]
        ob_map = observations[:, :11*15*10].view(batch, 11, 15, 10)
        ob_player = observations[:, 11*15*10:-10]
        ob_reward = observations[:, -10:]
        #print([torch.max(ob_map[:, :, :, i]) for i in range(10)])
        ob_map = torch.cat([torch.nn.functional.one_hot(
            ob_map[:, :, :, i].long(), self.factors[i]) for i in range(10)], dim=-1)

        #x = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
        #with torch.no_grad():
        #    ob_map = self.decompressor(x['map']).float()
        #player = x['player']

        #player = self.player_proj(player.float() / 99)
        #player, _ = self.lstm(player)
        #return player, None

        ob_player = self.player_encoder(ob_player.float() / 99)
        #ob_player = self.player_proj(player.float()/99)
        #return ob_player, None

        #ob_map = observations[:, :(11*15)].view(batch, 11, 15)
        #ob_map = self.decompressor(ob_map).float()
        ob_map = self.map_2d(ob_map.permute(0, 3, 1, 2).float())

        reward = observations[:, -10:].float() / 10000
        ob_reward = self.reward_proj(reward)

        ob = torch.cat([ob_map, ob_player, ob_reward], dim=1)
        #ob = F.relu(ob)
        return ob, None
        return self.proj(ob), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        #action = [self.actor(flat_hidden)]
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
