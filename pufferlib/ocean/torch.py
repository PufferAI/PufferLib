from torch import nn
import torch
import torch.nn.functional as F

from functools import partial
import pufferlib.models

from pufferlib.models import Default as Policy
from pufferlib.models import Convolutional as Conv
Recurrent = pufferlib.models.LSTMWrapper
import numpy as np

class NMMO3LSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=256, hidden_size=256, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

class NMMO3(nn.Module):
    def __init__(self, env, hidden_size=256, output_size=256):
        super().__init__()
        #self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.num_actions = env.single_action_space.n
        self.factors = np.array([4, 4, 17, 5, 3, 5, 5, 5, 7, 4])
        self.offsets = torch.tensor([0] + list(np.cumsum(self.factors)[:-1])).cuda().view(1, -1, 1, 1)
        self.cum_facs = np.cumsum(self.factors)

        self.multihot_dim = self.factors.sum()

        self.map_2d = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(self.multihot_dim, 64, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.Flatten(),
        )

        self.player_discrete_encoder = nn.Sequential(
            nn.Embedding(128, 32),
            nn.Flatten(),
        )

        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(1689, hidden_size)),
            nn.ReLU(),
        )

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

        map_buf = torch.zeros(batch, self.multihot_dim, 11, 15, device=ob_map.device, dtype=torch.float32)
        codes = ob_map.permute(0, 3, 1, 2) + self.offsets
        map_buf.scatter_(1, codes, 1)
        ob_map = self.map_2d(map_buf)

        player_discrete = self.player_discrete_encoder(ob_player.int())

        obs = torch.cat([ob_map, player_discrete, ob_player.float(), ob_reward], dim=1)
        obs = self.proj(obs)
        return obs, None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class Snake(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(8, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

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

class Grid(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(32, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.Flatten(),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size)),
            nn.ReLU(),
        )

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))
        else:
            num_actions = env.single_action_space.n
            self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, num_actions), std=0.01)

        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        hidden = observations.view(-1, 11, 11).long()
        hidden = F.one_hot(hidden, 32).permute(0, 3, 1, 2).float()
        hidden = self.network(hidden)
        return hidden, None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        value = self.value_fn(flat_hidden)
        if self.is_continuous:
            mean = self.decoder_mean(flat_hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            probs = torch.distributions.Normal(mean, std)
            batch = flat_hidden.shape[0]
            return probs, value
        else:
            action = self.actor(flat_hidden)
            return action, value

class Go(nn.Module):
    def __init__(self, env, cnn_channels=64, hidden_size=128, **kwargs):
        super().__init__()
        # 3 categories 2 boards. 
        # categories = player, opponent, empty
        # boards = current, previous
        self.cnn = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(2, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride = 1)),
            nn.Flatten(),
        )

        obs_size = env.single_observation_space.shape[0]
        self.grid_size = int(np.sqrt((obs_size-2)/2))
        output_size = self.grid_size - 4
        cnn_flat_size = cnn_channels * output_size * output_size
        
        self.flat = pufferlib.pytorch.layer_init(nn.Linear(2,32))
        
        self.proj = pufferlib.pytorch.layer_init(nn.Linear(cnn_flat_size + 32, hidden_size))

        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std=0.01)

        self.value_fn = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1), std=1)
   
    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        grid_size = int(np.sqrt((observations.shape[1] - 2) / 2))
        full_board = grid_size * grid_size 
        black_board = observations[:, :full_board].view(-1,1, grid_size,grid_size).float()
        white_board = observations[:, full_board:-2].view(-1,1, grid_size, grid_size).float()
        board_features = torch.cat([black_board, white_board],dim=1)
        flat_feature1 = observations[:, -2].unsqueeze(1).float()
        flat_feature2 = observations[:, -1].unsqueeze(1).float()
        # Pass board through cnn
        cnn_features = self.cnn(board_features)
        # Pass extra feature
        flat_features = torch.cat([flat_feature1, flat_feature2],dim=1)
        flat_features = self.flat(flat_features)
        # pass all features
        features = torch.cat([cnn_features, flat_features], dim=1)
        features = F.relu(self.proj(features))

        return features, None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        value = self.value_fn(flat_hidden)
        action = self.actor(flat_hidden)
        return action, value
    
class MOBA(nn.Module):
    def __init__(self, env, cnn_channels=128, hidden_size=128, **kwargs):
        super().__init__()
        self.cnn = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(16 + 3, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.Flatten(),
        )
        self.flat = pufferlib.pytorch.layer_init(nn.Linear(26, 128))
        self.proj = pufferlib.pytorch.layer_init(nn.Linear(128+cnn_channels, hidden_size))

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()
            self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)

        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        cnn_features = observations[:, :-26].view(-1, 11, 11, 4).long()
        if cnn_features[:, :, :, 0].max() > 15:
            print('Invalid map value:', cnn_features[:, :, :, 0].max())
            breakpoint()
            exit(1)
        map_features = F.one_hot(cnn_features[:, :, :, 0], 16).permute(0, 3, 1, 2).float()
        extra_map_features = (cnn_features[:, :, :, -3:].float() / 255).permute(0, 3, 1, 2)
        cnn_features = torch.cat([map_features, extra_map_features], dim=1)
        #print('observations 2d: ', map_features[0].cpu().numpy().tolist())
        cnn_features = self.cnn(cnn_features)
        #print('cnn features: ', cnn_features[0].detach().cpu().numpy().tolist())

        flat_features = observations[:, -26:].float() / 255.0
        #print('observations 1d: ', flat_features[0, 0])
        flat_features = self.flat(flat_features)
        #print('flat features: ', flat_features[0].detach().cpu().numpy().tolist())

        features = torch.cat([cnn_features, flat_features], dim=1)
        features = F.relu(self.proj(F.relu(features)))
        #print('features: ', features[0].detach().cpu().numpy().tolist())
        return features, None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        #print('lstm: ', flat_hidden[0].detach().cpu().numpy().tolist())
        value = self.value_fn(flat_hidden)
        if self.is_continuous:
            mean = self.decoder_mean(flat_hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            probs = torch.distributions.Normal(mean, std)
            batch = flat_hidden.shape[0]
            return probs, value
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)

            #argmax_samples = [torch.argmax(a, dim=1).detach().cpu().numpy().tolist() for a in action]
            #print('argmax samples: ', argmax_samples)

            return action, value

class TrashPickup(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(5, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        observations = observations.view(-1, 5, 11, 11).float()
        return self.network(observations), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class TowerClimbLSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size = 256, hidden_size = 256, num_layers = 1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

class TowerClimb(nn.Module):
    def __init__(self, env, cnn_channels=16, hidden_size = 256, **kwargs):
        super().__init__()
        self.network = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Conv3d(1, cnn_channels, 3, stride = 1)),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(
                    nn.Conv3d(cnn_channels, cnn_channels, 3, stride=1)),
                nn.Flatten()       
        )
        cnn_flat_size = cnn_channels * 1 * 1 * 5

        # Process player obs
        self.flat = pufferlib.pytorch.layer_init(nn.Linear(3,16))

        # combine
        self.proj = pufferlib.pytorch.layer_init(
                nn.Linear(cnn_flat_size + 16, hidden_size))
        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std = 0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1 ), std=1)

    def forward(self, observations, state=None):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value, state
    def encode_observations(self, observations):
        board_state = observations[:,:225]
        player_info = observations[:, -3:] 
        board_features = board_state.view(-1, 1, 5,5,9).float()
        cnn_features = self.network(board_features)
        flat_features = self.flat(player_info.float())
        
        features = torch.cat([cnn_features,flat_features],dim = 1)
        features = self.proj(features)
        return features, None
    
    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        
        return action, value

