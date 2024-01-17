from pdb import set_trace as T

import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.emulation
import pufferlib.models
import pufferlib.pytorch
from pufferlib.environments import try_import

try_import("nmmo")
from nmmo.entity.entity import EntityState


class Recurrent(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, input_size=256, hidden_size=256, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

class Policy(pufferlib.models.Policy):
  NUM_ATTRS = 34
  EntityId = EntityState.State.attr_name_to_col["id"]
  tile_offset = torch.tensor([i*256 for i in range(3)])
  entity_offset = torch.tensor([i*256 for i in range(3, 34)])

  def __init__(self, env, input_size=256, hidden_size=256, output_size=256):
      super().__init__(env)

      self.flat_observation_space = env.flat_observation_space
      self.flat_observation_structure = env.flat_observation_structure

      # A dumb example encoder that applies a linear layer to agent self features
      self.embedding = torch.nn.Embedding(self.NUM_ATTRS*256, 32)
      self.tile_conv_1 = torch.nn.Conv2d(96, 32, 3)
      self.tile_conv_2 = torch.nn.Conv2d(32, 8, 3)
      self.tile_fc = torch.nn.Linear(8*11*11, input_size)

      self.entity_fc = torch.nn.Linear(31*32, input_size)

      self.proj_fc = torch.nn.Linear(2*input_size, input_size)

      self.decoders = torch.nn.ModuleList([torch.nn.Linear(hidden_size, n)
              for n in env.single_action_space.nvec])
      self.value_head = torch.nn.Linear(hidden_size, 1)

  def encode_observations(self, env_outputs):
    env_outputs = pufferlib.emulation.unpack_batched_obs(env_outputs,
        self.flat_observation_space, self.flat_observation_structure)

    tile = env_outputs['Tile']
    # Center on player
    # This is cursed without clone??
    tile[:, :, :2] -= tile[:, 112:113, :2].clone() 
    tile[:, :, :2] += 7
    tile = self.embedding(
        tile.long().clip(0, 255) + self.tile_offset.to(tile.device)
    )

    agents, tiles, features, embed = tile.shape
    tile = tile.view(agents, tiles, features*embed).transpose(1, 2).view(agents, features*embed, 15, 15)

    tile = self.tile_conv_1(tile)
    tile = F.relu(tile)
    tile = self.tile_conv_2(tile)
    tile = F.relu(tile)
    tile = tile.contiguous().view(agents, -1)
    tile = self.tile_fc(tile)
    tile = F.relu(tile)

    # Pull out rows corresponding to the agent
    agentEmb = env_outputs["Entity"]
    my_id = env_outputs["AgentId"][:,0]
    entity_ids = agentEmb[:,:,self.EntityId]
    mask = (entity_ids == my_id.unsqueeze(1)) & (entity_ids != 0)
    mask = mask.int()
    row_indices = torch.where(mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1)))
    entity = agentEmb[torch.arange(agentEmb.shape[0]), row_indices]

    entity = self.embedding(
        entity.long().clip(0, 255) + self.entity_offset.to(entity.device)
    )
    agents, attrs, embed = entity.shape
    entity = entity.view(agents, attrs*embed)

    entity = self.entity_fc(entity)
    entity = F.relu(entity)

    obs = torch.cat([tile, entity], dim=-1)
    return self.proj_fc(obs), None

  def decode_actions(self, hidden, lookup, concat=True):
      value = self.value_head(hidden)
      actions = [dec(hidden) for dec in self.decoders]
      return actions, value
