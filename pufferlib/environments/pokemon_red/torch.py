from torch.nn import functional as F
from pdb import set_trace as T
import pufferlib.models
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, is_image_space, get_flattened_obs_dim, NatureCNN, TensorDict, gym
from gymnasium import spaces
import torch as th
from torch import nn
import pufferlib.emulation
import numpy as np

class Recurrent(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

# class Policy(pufferlib.models.Convolutional):
#     def __init__(self, env, input_size=512, hidden_size=512, output_size=512,
#             framestack=3, flat_size=64*5*6):
#         super().__init__(
#             env=env,
#             input_size=input_size,
#             hidden_size=hidden_size,
#             output_size=output_size,
#             framestack=framestack,
#             flat_size=flat_size,
#             channels_last=True,
#         )


class Policy(pufferlib.models.Policy):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        env: spaces.Dict,
        cnn_output_dim: int = 256*2,
        normalized_image: bool = False,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(env)

        # observation_space.spaces.items()

        # image (3, 36, 40)
        # self.image_cnn = NatureCNN(observation_space['image'], features_dim=cnn_output_dim, normalized_image=normalized_image)
        # nature cnn (4, 36, 40), output_dim = 512 cnn_output_dim
        # if not isinstance(env.observation_space, spaces.Dict):
        #     raise TypeError("Expected env.observation_space to be a gym.spaces.Dict")

        n_input_channels = env.structured_observation_space['image'].shape[0]
        '''
        (Pdb) dir(env)
['__annotations__', '__class__', '__class_getitem__', '__delattr__', '__dict__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__orig_bases__', '__parameters__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_is_protocol', '_np_random', 'action_space', 'box_observation_space', 'close', 'done', 'env', 'flat_action_space', 'flat_action_structure', 'flat_observation_space', 'flat_observation_structure', 'get_wrapper_attr', 'initialized', 'is_action_checked', 'is_observation_checked', 'metadata', 'multidiscrete_action_space', 'np_random', 'observation_space', 'pad_observation', 'postprocessor', 'render', 'render_mode', 'render_modes', 'reset', 'reward_range', 'seed', 'spec', 'step', 'structured_action_space', 'structured_observation_space', 'unpack_batched_obs', 'unwrapped']
(Pdb) env.render
<bound method GymnasiumPufferEnv.render of <pufferlib.emulation.GymnasiumPufferEnv object at 0x7fb43917df30>>
(Pdb) env.observation_space
Box(-3.4028235e+38, 3.4028235e+38, (19591,), float32)
(Pdb) env.observation_space[0]
*** TypeError: 'Box' object is not subscriptable
(Pdb) env.observation_space
Box(-3.4028235e+38, 3.4028235e+38, (19591,), float32)
(Pdb) 
        '''

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32*2, kernel_size=8, stride=4, padding=(2, 0)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(9, 9)),
            nn.Conv2d(32*2, 64*2, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64*2, 64*2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(env.structured_observation_space['image'].sample()[None]).float()).shape[1]

        self.cnn_linear = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.ReLU())


        # sprite embedding
        sprite_emb_dim = 8
        # minimap_sprite id use embedding (9, 10) -> (9, 10, 8)
        self.minimap_sprite_embedding = nn.Embedding(390, sprite_emb_dim, padding_idx=0)
        # change to channel first (8, 9, 10) with permute in forward()

        # warp embedding
        warp_emb_dim = 8
        # minimap_warp id use embedding (9, 10) -> (9, 10, 8)
        self.minimap_warp_embedding = nn.Embedding(830, warp_emb_dim, padding_idx=0)

        # minimap (14 + 8 + 8, 9, 10)
        n_input_channels = env.structured_observation_space['minimap'].shape[0] + sprite_emb_dim + warp_emb_dim
        self.minimap_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32*2, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32*2, 64*2, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64*2, 128*2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # # Compute shape by doing one forward pass
        # with th.no_grad():
        #     n_flatten = self.minimap_cnn(th.as_tensor(observation_space['minimap'].sample()[None]).float()).shape[1]
        n_flatten = 128*2*2
        self.minimap_cnn_linear = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.ReLU())

        # poke_move_ids (12, 4) -> (12, 4, 8)
        self.poke_move_ids_embedding = nn.Embedding(167, 8, padding_idx=0)
        # concat with poke_move_pps (12, 4, 2)
        # input (12, 4, 10) for fc relu
        self.move_fc_relu = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        )
        # max pool
        self.move_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))
        # output (12, 1, 16), sqeeze(-2) -> (12, 16)

        # poke_type_ids (12, 2) -> (12, 2, 8)
        self.poke_type_ids_embedding = nn.Embedding(17, 8, padding_idx=0)  # change to 18
        # (12, 2, 8) -> (12, 8) by sum(dim=-2)

        # poke_ids (12, ) -> (12, 8)
        self.poke_ids_embedding = nn.Embedding(192, 16, padding_idx=0)
        
        # pokemon fc relu
        self.poke_fc_relu = nn.Sequential(
            nn.Linear(63, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # pokemon party head
        self.poke_party_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        # get the first 6 pokemon and do max pool
        self.poke_party_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 64))

        # pokemon opp head
        self.poke_opp_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        # get the last 6 pokemon and do max pool
        self.poke_opp_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 64))

        # item_ids embedding
        self.item_ids_embedding = nn.Embedding(256, 32, padding_idx=0)  # (20, 32)
        # item_ids fc relu
        self.item_ids_fc_relu = nn.Sequential(
            nn.Linear(33, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        # item_ids max pool
        self.item_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 32))

        # event_ids embedding
        self.event_ids_embedding = nn.Embedding(2570, 64, padding_idx=0)  # (20, )
        # event_ids fc relu
        self.event_ids_fc_relu = nn.Sequential(
            nn.Linear(65, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # event_ids max pool
        self.event_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 64))

        map_ids_emb_dim = 32
        # map_ids embedding
        self.map_ids_embedding = nn.Embedding(256, map_ids_emb_dim, padding_idx=0)
        # map_ids fc relu
        self.map_ids_fc_relu = nn.Sequential(
            nn.Linear(33, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        # map_ids max pool
        self.map_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, map_ids_emb_dim))

        # self._features_dim = 410 + 256 + map_ids_emb_dim
        self._features_dim = 579 + 256 + map_ids_emb_dim + 512

        # breakpoint()

        self.flat_observation_space = env.flat_observation_space
        self.flat_observation_structure = env.flat_observation_structure
        # observation_space.spaces.items()
        
        # image (3, 36, 40)
        self.image_cnn = NatureCNN(env.structured_observation_space['image'], features_dim=cnn_output_dim, normalized_image=normalized_image)

        # poke_move_ids (12, 4) -> (12, 4, 8)
        self.poke_move_ids_embedding = nn.Embedding(167, 8, padding_idx=0)
        # concat with poke_move_pps (12, 4, 2)
        # input (12, 4, 10) for fc relu
        self.move_fc_relu = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        )
        # max pool
        self.move_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))
        # output (12, 1, 16), sqeeze(-2) -> (12, 16)

        # poke_type_ids (12, 2) -> (12, 2, 8)
        self.poke_type_ids_embedding = nn.Embedding(17, 8, padding_idx=0)
        # (12, 2, 8) -> (12, 8) by sum(dim=-2)

        # poke_ids (12, ) -> (12, 8)
        self.poke_ids_embedding = nn.Embedding(192, 16, padding_idx=0)
        
        # pokemon fc relu
        self.poke_fc_relu = nn.Sequential(
            nn.Linear(63, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # pokemon party head
        self.poke_party_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        # get the first 6 pokemon and do max pool
        self.poke_party_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 32))

        # pokemon opp head
        self.poke_opp_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        # get the last 6 pokemon and do max pool
        self.poke_opp_head_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 32))

        # item_ids embedding
        self.item_ids_embedding = nn.Embedding(256, 16, padding_idx=0)  # (20, 16)
        # item_ids fc relu
        self.item_ids_fc_relu = nn.Sequential(
            nn.Linear(17, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        # item_ids max pool
        self.item_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))

        # event_ids embedding
        self.event_ids_embedding = nn.Embedding(2570, 16, padding_idx=0)  # (20, )
        # event_ids fc relu
        self.event_ids_fc_relu = nn.Sequential(
            nn.Linear(17, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        # event_ids max pool
        self.event_ids_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 16))

        self._features_dim = 1251
        # self._features_dim = 406

        # self.fc1 = nn.Linear(406,512)
        self.fc1 = nn.Linear(1251,512)
        self.fc2 = nn.Linear(512,512)
        self.action = nn.Linear(512, self.action_space.n)
        self.value_head = nn.Linear(512,1)
        

    # breakpoint()
    def encode_observations(self, observations: TensorDict) -> th.Tensor:
        # sz = [
        #     int(np.prod(subspace.shape))
        #     for subspace in self.flat_observation_space.values()
        # ]
        observations = pufferlib.emulation.unpack_batched_obs(observations,
        self.flat_observation_space, self.flat_observation_structure)
        
        # img = self.image_cnn(observations['image'])  # (256, )
        img = self.cnn_linear(self.cnn(observations['image']))  # (512, )
        
        
        # minimap_sprite
        minimap_sprite = observations['minimap_sprite'].to(th.int)  # (9, 10)
        embedded_minimap_sprite = self.minimap_sprite_embedding(minimap_sprite)  # (9, 10, 8)
        embedded_minimap_sprite = embedded_minimap_sprite.permute(0, 3, 1, 2)  # (B, 8, 9, 10)
        # minimap_warp
        minimap_warp = observations['minimap_warp'].to(th.int)  # (9, 10)
        embedded_minimap_warp = self.minimap_warp_embedding(minimap_warp)  # (9, 10, 8)
        embedded_minimap_warp = embedded_minimap_warp.permute(0, 3, 1, 2)  # (B, 8, 9, 10)
        # concat with minimap
        minimap = observations['minimap']  # (14, 9, 10)
        minimap = th.cat([minimap, embedded_minimap_sprite, embedded_minimap_warp], dim=1)  # (14 + 8 + 8, 9, 10)
        # minimap
        minimap = self.minimap_cnn_linear(self.minimap_cnn(minimap))  # (256, )

        
        
        # Pokemon
        # Moves
        embedded_poke_move_ids = self.poke_move_ids_embedding(observations['poke_move_ids'].to(th.int))
        poke_move_pps = observations['poke_move_pps']
        poke_moves = th.cat([embedded_poke_move_ids, poke_move_pps], dim=-1)
        poke_moves = self.move_fc_relu(poke_moves)
        poke_moves = self.move_max_pool(poke_moves).squeeze(-2)  # (12, 16)
        # Types
        embedded_poke_type_ids = self.poke_type_ids_embedding(observations['poke_type_ids'].to(th.int))
        poke_types = th.sum(embedded_poke_type_ids, dim=-2)  # (12, 8)
        # Pokemon ID
        embedded_poke_ids = self.poke_ids_embedding(observations['poke_ids'].to(th.int))
        poke_ids = embedded_poke_ids  # (12, 8)
        # Pokemon stats (12, 23)
        poke_stats = observations['poke_all']
        # All pokemon features
        pokemon_concat = th.cat([poke_moves, poke_types, poke_ids, poke_stats], dim=-1)  # (12, 63)
        pokemon_features = self.poke_fc_relu(pokemon_concat)  # (12, 32)

        # Pokemon party head
        party_pokemon_features = pokemon_features[..., :6, :]  # (6, 32), ... for batch dim
        poke_party_head = self.poke_party_head(party_pokemon_features)  # (6, 32)
        poke_party_head = self.poke_party_head_max_pool(poke_party_head).squeeze(-2)  # (6, 32) -> (32, )

        # Pokemon opp head
        opp_pokemon_features = pokemon_features[..., 6:, :]  # (6, 32), ... for batch dim
        poke_opp_head = self.poke_opp_head(opp_pokemon_features)  # (6, 32)
        poke_opp_head = self.poke_opp_head_max_pool(poke_opp_head).squeeze(-2)  # (6, 32) -> (32, )


        # Items
        embedded_item_ids = self.item_ids_embedding(observations['item_ids'].to(th.int))  # (20, 16)
        # item_quantity
        item_quantity = observations['item_quantity']  # (20, 1)
        item_concat = th.cat([embedded_item_ids, item_quantity], dim=-1)  # (20, 33)
        item_features = self.item_ids_fc_relu(item_concat)  # (20, 32)
        item_features = self.item_ids_max_pool(item_features).squeeze(-2)  # (20, 32) -> (32, )

        # # Items
        # embedded_item_ids = self.item_ids_embedding(observations['item_ids'].to(th.int))  # (20, 16)
        # # item_quantity
        # item_quantity = observations['item_quantity']  # (20, 1)
        # item_concat = th.cat([embedded_item_ids, item_quantity], dim=-1)  # (20, 17)
        # item_features = self.item_ids_fc_relu(item_concat)  # (20, 16)
        # item_features = self.item_ids_max_pool(item_features).squeeze(-2)  # (20, 16) -> (16, )

        # Events
        embedded_event_ids = self.event_ids_embedding(observations['event_ids'].to(th.int))
        # event_step_since
        event_step_since = observations['event_step_since']  # (20, 1)
        event_concat = th.cat([embedded_event_ids, event_step_since], dim=-1)  # (20, 17)
        event_features = self.event_ids_fc_relu(event_concat)
        event_features = self.event_ids_max_pool(event_features).squeeze(-2)  # (20, 16) -> (16, )
        
        # Maps
        embedded_map_ids = self.map_ids_embedding(observations['map_ids'].to(th.int))  # (20, 16)
        # map_step_since
        map_step_since = observations['map_step_since']  # (20, 1)
        map_concat = th.cat([embedded_map_ids, map_step_since], dim=-1)  # (20, 17)
        map_features = self.map_ids_fc_relu(map_concat)  # (20, 16)
        map_features = self.map_ids_max_pool(map_features).squeeze(-2)  # (20, 16) -> (16, )

        # Raw vector
        vector = observations['vector']  # (99, )

        # # Raw vector
        # vector = observations['vector']  # (54, )

        # Concat all features

        # Concat all features
        all_features = th.cat([img, minimap, poke_party_head, poke_opp_head, item_features, event_features, vector, map_features], dim=-1)  # (410 + 256, )
        hidden = self.fc2(F.relu(self.fc1(all_features)))
        return hidden, None
        
    def decode_actions(self, hidden, lookup):
        action = self.action(hidden)
        value = self.value_head(hidden)
        return action, value