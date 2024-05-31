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
    def __init__(self, inference_batch_size, train_batch_size):
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

        #codes = torch.randint(0, 4*4*16*5*3*5*5*6*7*4, (agents, 11, 15)).cuda()
        #obs_view = obs.view(agents, n_channels, 11, 15)

        obs_inference = torch.zeros(inference_batch_size, 59, 11*15)
        obs_inference_view = obs_inference.view(inference_batch_size, 59, 11, 15)
        obs_train = torch.zeros(train_batch_size, 59, 11*15)
        obs_train_view = obs_train.view(train_batch_size, 59, 11, 15)

        self.register_buffer('obs_inference', obs_inference)
        self.register_buffer('obs_train', obs_train)
        self.register_buffer('obs_inference_view', obs_inference_view)
        self.register_buffer('obs_train_view', obs_train_view)

    def forward(self, codes):
        batch = codes.shape[0]

        #obs = torch.zeros(batch, 59, 11*15, device=codes.device)
        #obs_view = obs.view(batch, 59, 11, 15)

        if batch == self.obs_inference.shape[0]:
            obs = self.obs_inference
            obs_view = self.obs_inference_view
        elif batch == self.obs_train.shape[0]:
            obs = self.obs_train
            obs_view = self.obs_train_view
        else:
            raise ValueError('Invalid batch size')
        obs.fill_(0)

        codes = codes.view(codes.shape[0], 1, -1)
        dec = self.add + (codes//self.div) % self.factors
        obs.scatter_(1, dec, 1)
        return obs_view


class Policy(nn.Module):
    def __init__(self, env, inference_batch_size, train_batch_size):
        super().__init__()
        self.emulated = env.emulated
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        #self.env = env

        self.num_actions = env.single_action_space.n
        #self.num_actions = self.action_space.n
        #self.num_actions = self.action_space.shape[0]
        hidden_size = 256
        output_size = 256

        self.decompressor = Decompressor(inference_batch_size, train_batch_size)

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

        #self.ob_map = torch.zeros(100000, 59, 11, 15, device='cuda')
        #self.players = torch.zeros(100000, 44, device='cuda').int()

        #self.player_emb = torch.zeros(100000, 128, device='cuda')
        #self.map_emb = torch.zeros(100000, 128, device='cuda')

        ob_map = torch.zeros(inference_batch_size, 59, 11, 15, device='cuda')
        players = torch.zeros(inference_batch_size, 44, device='cuda').int()

        self.register_buffer('ob_map', ob_map)
        self.register_buffer('players', players)

    def forward(self, x):
        hidden, lookup = self.encode_observations(x)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations, unflatten=False):
        batch = observations.shape[0]
        x = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
        #x = pufferlib.pytorch.nativize_observation(observations, self.emulated)

        with torch.no_grad():
            ob_map = self.decompressor(x['map']).float()
        #    #ob_map = decode_map(x['map']).float()

        player = x['player']

        #env_ob = self.env.buf.observations
        #assert env_ob == observations.cpu().numpy()

        #ob_map = self.ob_map[:batch]
        #player = self.players[:batch]

        #ob_map = self.ob_map
        #player = self.players

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
