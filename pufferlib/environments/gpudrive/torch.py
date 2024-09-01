from torch import nn
import torch
import torch.nn.functional as F

from functools import partial
import pufferlib.models

from pufferlib.models import Default as Policy
Recurrent = pufferlib.models.LSTMWrapper

EGO_STATE_DIM = 6
PARTNER_DIM = 10
ROAD_MAP_DIM = 13

MAX_CONTROLLED_VEHICLES = 32
ROADMAP_AGENT_FEAT_DIM = MAX_CONTROLLED_VEHICLES - 1
TOP_K_ROADPOINTS = 64 # Number of visible roadpoints from the road graph

def unpack_obs(obs_flat):
    """
    Unpack the flattened observation into the ego state and visible state.
    Args:
        obs_flat (torch.Tensor): flattened observation tensor of shape (batch_size, obs_dim)
    Return:
        ego_state, road_objects, stop_signs, road_graph (torch.Tensor).
    """
    # Unpack ego and visible state
    ego_state = obs_flat[:, :EGO_STATE_DIM]
    vis_state = obs_flat[:, EGO_STATE_DIM :]
                                                                                                                   # Visible state object order: road_objects, road_points
    # Find the ends of each section
    ro_end_idx = PARTNER_DIM * ROADMAP_AGENT_FEAT_DIM
    rg_end_idx = ro_end_idx + (ROAD_MAP_DIM * TOP_K_ROADPOINTS)
    
    # Unflatten and reshape to (batch_size, num_objects, object_dim)
    road_objects = (vis_state[:, :ro_end_idx]).reshape(
        -1, ROADMAP_AGENT_FEAT_DIM, PARTNER_DIM
    )
    road_graph = (vis_state[:, ro_end_idx:rg_end_idx]).reshape(
        -1,
        TOP_K_ROADPOINTS,
        ROAD_MAP_DIM,
    )
    return ego_state, road_objects, road_graph

class Policy(nn.Module):
    def __init__(self, env, input_size=64, hidden_size=128, **kwargs):
        super().__init__()
        self.ego_embed = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(EGO_STATE_DIM, input_size)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.partner_embed = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(PARTNER_DIM, input_size)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.road_map_embed = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(ROAD_MAP_DIM, input_size)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.proj = pufferlib.pytorch.layer_init(nn.Linear(3*input_size, hidden_size))

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        ego_state, road_objects, road_graph = unpack_obs(observations)
        ego_embed = self.ego_embed(ego_state)
        partner_embed, _ = self.partner_embed(road_objects).max(dim=1)
        road_map_embed, _ = self.road_map_embed(road_graph).max(dim=1)
        embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)
        return self.proj(embed), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
