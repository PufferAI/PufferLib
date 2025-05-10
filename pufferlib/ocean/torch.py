from typing import Any, Tuple

from gymnasium import spaces

from torch import nn
import torch
from torch.distributions.normal import Normal
from torch import nn
import torch.nn.functional as F

import pufferlib
import pufferlib.models

from pufferlib.models import Default as Policy
from pufferlib.models import Convolutional as Conv
Recurrent = pufferlib.models.LSTMWrapper
from pufferlib.pytorch import layer_init, _nativize_dtype, nativize_tensor
import numpy as np


class NMMO3LSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512):
        super().__init__(env, policy, input_size, hidden_size)

class NMMO3(nn.Module):
    def __init__(self, env, hidden_size=512, output_size=512, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        #self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.num_actions = env.single_action_space.n
        self.factors = np.array([4, 4, 17, 5, 3, 5, 5, 5, 7, 4])
        self.offsets = torch.tensor([0] + list(np.cumsum(self.factors)[:-1])).cuda().view(1, -1, 1, 1)
        self.cum_facs = np.cumsum(self.factors)

        self.multihot_dim = self.factors.sum()
        self.is_continuous = False

        self.map_2d = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(self.multihot_dim, 256, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(256, 256, 3, stride=1)),
            nn.Flatten(),
        )

        self.player_discrete_encoder = nn.Sequential(
            nn.Embedding(128, 32),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(2073, hidden_size)),
            nn.ReLU(),
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, self.num_actions), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

        # Pre-allocate allows compilation
        map_buf = torch.zeros(32768, self.multihot_dim, 11, 15, dtype=torch.float32)
        self.register_buffer('map_buf', map_buf)

    def forward(self, x, state=None):
        hidden = self.encode_observations(x)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        batch = observations.shape[0]
        ob_map = observations[:, :11*15*10].view(batch, 11, 15, 10)
        ob_player = observations[:, 11*15*10:-10]
        ob_reward = observations[:, -10:]

        batch = ob_map.shape[0]
        map_buf = self.map_buf[:batch]
        map_buf.zero_()
        codes = ob_map.permute(0, 3, 1, 2) + self.offsets
        map_buf.scatter_(1, codes, 1)
        ob_map = self.map_2d(map_buf)

        player_discrete = self.player_discrete_encoder(ob_player.int())

        obs = torch.cat([ob_map, player_discrete, ob_player.float(), ob_reward], dim=1)
        obs = self.proj(obs)
        return obs

    def decode_actions(self, flat_hidden):
        flat_hidden = self.layer_norm(flat_hidden)
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class Snake(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128,
            use_p3o=False, p3o_horizon=32, use_diayn=False, diayn_skills=8, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.use_diayn = use_diayn

        encode_dim = cnn_channels
        if use_diayn:
            encode_dim += diayn_skills
            self.diayn_skills = diayn_skills
            '''
            self.diayn_discriminator = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Conv2d(64, cnn_channels, 5, stride=3)),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(
                    nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, diayn_skills)),
            )
            '''
            self.diayn_discriminator = nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(64*env.single_action_space.n, hidden_size)),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(nn.Linear(hidden_size, diayn_skills)),
            )
 
        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(8, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(encode_dim, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)

        self.use_p3o = use_p3o
        self.p3o_horizon = p3o_horizon
        if use_p3o:
            self.value_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, p3o_horizon), std=1)
            self.value_logstd = nn.Parameter(torch.zeros(1, p3o_horizon))
        else:
            self.value = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1), std=1)

    def discrim_forward(self, obs):
        #obs = F.one_hot(obs.long(), 8).permute(0, 1, 4, 2, 3).float()
        #B, f, c, h, w = obs.shape
        #obs = obs.reshape(B, f*c, h, w)
        return self.diayn_discriminator(obs)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        observations = F.one_hot(observations.long(), 8).permute(0, 3, 1, 2).float()
        hidden = self.network(observations)

        if self.use_diayn:
            z_one_hot = F.one_hot(state.diayn_z, self.diayn_skills).float()
            hidden = torch.cat([hidden, z_one_hot], dim=1)

        return self.proj(hidden)

    def decode_actions(self, hidden):
        action = self.actor(hidden)

        if self.use_p3o:
            value_mean = self.value_mean(hidden)
            value_logstd = self.value_logstd.expand_as(value_mean)
            return action, value_mean, value_logstd
        else:
            value = self.value(hidden)
            return action, value

class Grid(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
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

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        hidden = observations.view(-1, 11, 11).long()
        hidden = F.one_hot(hidden, 32).permute(0, 3, 1, 2).float()
        hidden = self.network(hidden)
        return hidden

    def decode_actions(self, flat_hidden, state=None):
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
        self.hidden_size = hidden_size
        self.is_continuous = False
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
   
    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
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

        return features

    def decode_actions(self, flat_hidden, state=None):
        value = self.value_fn(flat_hidden)
        action = self.actor(flat_hidden)
        return action, value
    
class MOBA(nn.Module):
    def __init__(self, env, cnn_channels=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
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

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
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
        return features

    def decode_actions(self, flat_hidden):
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
        self.hidden_size = hidden_size
        self.is_continuous = False
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

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        observations = observations.view(-1, 5, 11, 11).float()
        return self.network(observations)

    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class TowerClimbLSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size = 256, hidden_size = 256):
        super().__init__(env, policy, input_size, hidden_size)

class TowerClimb(nn.Module):
    def __init__(self, env, cnn_channels=16, hidden_size = 256, **kwargs):
        self.hidden_size = hidden_size
        self.is_continuous = False
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
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        board_state = observations[:,:225]
        player_info = observations[:, -3:] 
        board_features = board_state.view(-1, 1, 5,5,9).float()
        cnn_features = self.network(board_features)
        flat_features = self.flat(player_info.float())
        
        features = torch.cat([cnn_features,flat_features],dim = 1)
        features = self.proj(features)
        return features
    
    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        
        return action, value


# ToDO: remove once Impulse Wars build is stable
try:
    from cy_impulse_wars import obsConstants
except:
    pass


class ImpulseWarsLSTM(Recurrent):
    def __init__(self, env: pufferlib.PufferEnv, policy: nn.Module, input_size: int = 256, hidden_size: int = 256):
        super().__init__(env, policy, input_size, hidden_size)


class ImpulseWarsPolicy(nn.Module):
    def __init__(
        self,
        env: pufferlib.PufferEnv,
        cnn_channels: int = 64,
        weapon_type_embedding_dims: int = 2,
        input_size: int = 256,
        hidden_size: int = 256,
        batch_size: int = 131_072,
        num_drones: int = 2,
        discretize_actions: bool = True,
        is_training: bool = True,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.is_continuous = not discretize_actions

        self.numDrones = num_drones
        self.isTraining = is_training
        self.obsInfo = obsConstants(self.numDrones)

        self.discreteFactors = np.array(
            [self.obsInfo.wallTypes] * self.obsInfo.numNearWallObs
            + [self.obsInfo.wallTypes + 1] * self.obsInfo.numFloatingWallObs
            + [self.numDrones + 1] * self.obsInfo.numProjectileObs,
        )
        discreteOffsets = torch.tensor([0] + list(np.cumsum(self.discreteFactors)[:-1]), device=device).view(
            1, -1
        )
        self.register_buffer("discreteOffsets", discreteOffsets, persistent=False)
        self.discreteMultihotDim = self.discreteFactors.sum()

        multihotBuffer = torch.zeros(batch_size, self.discreteMultihotDim, device=device)
        self.register_buffer("multihotOutput", multihotBuffer, persistent=False)

        # most of the observation is a 2D array of bytes, but the end
        # contains around 200 floats; this allows us to treat the end
        # of the observation as a float array
        _, *self.dtype = _nativize_dtype(
            np.dtype((np.uint8, (self.obsInfo.continuousObsBytes,))),
            np.dtype((np.float32, (self.obsInfo.continuousObsSize,))),
        )
        self.dtype = tuple(self.dtype)

        self.weaponTypeEmbedding = nn.Embedding(self.obsInfo.weaponTypes, weapon_type_embedding_dims)

        # each byte in the map observation contains 4 values:
        # - 2 bits for wall type
        # - 1 bit for is floating wall
        # - 1 bit for is weapon pickup
        # - 3 bits for drone index
        self.register_buffer(
            "unpackMask",
            torch.tensor([0x60, 0x10, 0x08, 0x07], dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer("unpackShift", torch.tensor([5, 4, 3, 0], dtype=torch.uint8), persistent=False)

        self.mapObsInputChannels = (self.obsInfo.wallTypes + 1) + 1 + 1 + self.numDrones
        self.mapCNN = nn.Sequential(
            layer_init(
                nn.Conv2d(
                    self.mapObsInputChannels,
                    cnn_channels,
                    kernel_size=5,
                    stride=3,
                )
            ),
            nn.ReLU(),
            layer_init(nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnnOutputSize = self._computeCNNShape()

        featuresSize = (
            cnnOutputSize
            + (self.obsInfo.numNearWallObs * (self.obsInfo.wallTypes + self.obsInfo.nearWallPosObsSize))
            + (
                self.obsInfo.numFloatingWallObs
                * (self.obsInfo.wallTypes + 1 + self.obsInfo.floatingWallInfoObsSize)
            )
            + (
                self.obsInfo.numWeaponPickupObs
                * (weapon_type_embedding_dims + self.obsInfo.weaponPickupPosObsSize)
            )
            + (
                self.obsInfo.numProjectileObs
                * (weapon_type_embedding_dims + self.obsInfo.projectileInfoObsSize + self.numDrones + 1)
            )
            + ((self.numDrones - 1) * (weapon_type_embedding_dims + self.obsInfo.enemyDroneObsSize))
            + (self.obsInfo.droneObsSize + weapon_type_embedding_dims)
            + self.obsInfo.miscObsSize
        )

        self.encoder = nn.Sequential(
            layer_init(nn.Linear(featuresSize, input_size)),
            nn.ReLU(),
        )

        if self.is_continuous:
            self.actorMean = layer_init(nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.actorLogStd = nn.Parameter(torch.zeros(1, env.single_action_space.shape[0]))
        else:
            self.actionDim = env.single_action_space.nvec.tolist()
            self.actor = layer_init(nn.Linear(hidden_size, sum(self.actionDim)), std=0.01)

        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, obs: torch.Tensor, state = None) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_observations(obs)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def unpack(self, batchSize: int, obs: torch.Tensor) -> torch.Tensor:
        # prepare map obs to be unpacked
        mapObs = obs[:, : self.obsInfo.mapObsSize].reshape((batchSize, -1, 1))
        # unpack wall types, weapon pickup types, and drone indexes
        mapObs = (mapObs & self.unpackMask) >> self.unpackShift
        # reshape so channels are first, required for torch conv2d
        return mapObs.permute(0, 2, 1).reshape(
            (batchSize, 4, self.obsInfo.mapObsRows, self.obsInfo.mapObsColumns)
        )

    def encode_observations(self, obs: torch.Tensor, state: Any = None) -> torch.Tensor:
        batchSize = obs.shape[0]

        mapObs = self.unpack(batchSize, obs)

        # one hot encode wall types
        wallTypeObs = mapObs[:, 0, :, :].long()
        wallTypes = F.one_hot(wallTypeObs, self.obsInfo.wallTypes + 1).permute(0, 3, 1, 2).float()

        # unsqueeze floating wall booleans (is wall a floating wall)
        floatingWallObs = mapObs[:, 1, :, :].unsqueeze(1)

        # unsqueeze map pickup booleans (does map tile contain a weapon pickup)
        mapPickupObs = mapObs[:, 2, :, :].unsqueeze(1)

        # one hot drone indexes
        droneIndexObs = mapObs[:, 3, :, :].long()
        droneIndexes = F.one_hot(droneIndexObs, self.numDrones).permute(0, 3, 1, 2).float()

        # combine all map observations and feed through CNN
        mapObs = torch.cat((wallTypes, floatingWallObs, mapPickupObs, droneIndexes), dim=1)
        map = self.mapCNN(mapObs)

        # process discrete observations
        multihotInput = (
            obs[:, self.obsInfo.nearWallTypesObsOffset : self.obsInfo.projectileTypesObsOffset]
            + self.discreteOffsets
        )
        multihotOutput = self.multihotOutput[:batchSize].zero_()
        multihotOutput.scatter_(1, multihotInput.long(), 1)

        weaponTypeObs = obs[:, self.obsInfo.projectileTypesObsOffset : self.obsInfo.discreteObsSize].int()
        weaponTypes = self.weaponTypeEmbedding(weaponTypeObs).float()
        weaponTypes = torch.flatten(weaponTypes, start_dim=1, end_dim=-1)

        # process continuous observations
        continuousObs = nativize_tensor(obs[:, self.obsInfo.continuousObsOffset :], self.dtype)
        # combine all observations and feed through final linear encoder
        features = torch.cat((map, multihotOutput, weaponTypes, continuousObs), dim=-1)

        return self.encoder(features)

    def decode_actions(self, hidden: torch.Tensor):
        if self.is_continuous:
            actionMean = self.actorMean(hidden)
            if self.isTraining:
                actionLogStd = self.actorLogStd.expand_as(actionMean)
                actionStd = torch.exp(actionLogStd)
                action = Normal(actionMean, actionStd)
            else:
                action = actionMean
        else:
            action = self.actor(hidden)
            action = torch.split(action, self.actionDim, dim=1)

        value = self.critic(hidden)

        return action, value

    def _computeCNNShape(self) -> int:
        mapSpace = spaces.Box(
            low=0,
            high=1,
            shape=(self.mapObsInputChannels, self.obsInfo.mapObsRows, self.obsInfo.mapObsColumns),
            dtype=np.float32,
        )

        with torch.no_grad():
            t = torch.as_tensor(mapSpace.sample()[None])
            return self.mapCNN(t).shape[1]

class GPUDrive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(6, input_size)),
            # nn.ReLU(),
            # pufferlib.pytorch.layer_init(
            #    nn.Linear(input_size, input_size))
        )
        max_road_objects = 13
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(max_road_objects, input_size)),
            # nn.ReLU(),
            # pufferlib.pytorch.layer_init(
            #    nn.Linear(input_size, input_size))
        )
        max_partner_objects = 7
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(max_partner_objects, input_size)),
            # nn.ReLU(),
            # pufferlib.pytorch.layer_init(
            #    nn.Linear(input_size, input_size))
        )

        '''
        self.post_mask_road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(input_size, input_size)),
        )
        self.post_mask_partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(input_size, input_size)),
        )
        '''
        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3*input_size,  hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        self.atn_dim = env.single_action_space.nvec.tolist()
        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, sum(self.atn_dim)), std = 0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1 ), std=1)
    
    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)
   
    def encode_observations(self, observations, state=None):
        ego_dim = 6
        partner_dim = 63 * 7
        road_dim = 200*7
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim:ego_dim+partner_dim]
        road_obs = observations[:, ego_dim+partner_dim:ego_dim+partner_dim+road_dim]
        
        partner_objects = partner_obs.view(-1, 63, 7)
        road_objects = road_obs.view(-1, 200, 7)
        road_continuous = road_objects[:, :, :6]  # First 6 features
        road_categorical = road_objects[:, :, 6]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)  # Shape: [batch, 200, 7]
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)
        #partner_features_post_mask = self.post_mask_partner_encoder(partner_features)
        #road_features_post_mask = self.post_mask_road_encoder(road_features)
        
        #concat_features = torch.cat([ego_features, road_features_post_mask, partner_features_post_mask], dim=1)
        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)
        
        # Apply max pooling across concatenated features
        # Reshape to [batch, 3, hidden_size] to pool across the 3 modalities
        # pooled_features = torch.max(
        #     concat_features.view(-1, 3, self.hidden_size), dim=1
        # )[0]
        
        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        # embedding = self.shared_embedding(concat_features)
        return embedding
    
    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        action = torch.split(action, self.atn_dim, dim=1)
        value = self.value_fn(flat_hidden)
        return action, value
