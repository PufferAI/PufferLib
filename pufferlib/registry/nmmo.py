from pdb import set_trace as T

import numpy as np
import torch
import torch.nn.functional as F


import pufferlib
import pufferlib.emulation
import pufferlib.models

from nmmo.entity.entity import EntityState


NUM_ATTRS = 26
EntityId = EntityState.State.attr_name_to_col["id"]
tile_offset = torch.tensor([i*256 for i in range(3)])
entity_offset = torch.tensor([i*256 for i in range(3, 26)])

def make_binding():
    '''Neural MMO binding creation function'''
    try:
        import nmmo
    except:
        raise pufferlib.utils.SetupError('Neural MMO (nmmo)')
    else:
        return pufferlib.emulation.Binding(
            env_cls=nmmo.Env,
            env_name='Neural MMO',
        )

class PufferNMMO(pufferlib.new_emulation.PufferEnv):
    def __init__(self, teams, postprocessor_cls, *args, **kwargs):
        try:
            import nmmo
        except:
            raise pufferlib.utils.SetupError('Neural MMO (nmmo)')
 
        self.env = nmmo.Env(*args, **kwargs)
        self.teams = teams

        self.postprocessors = {team: postprocessor_cls(team) for team in teams}

    def observation_space(self, team):
        '''Returns the observation space for a single agent'''
        raw_obs_space = {agent: self.env.observation_space(agent) for agent in team}
        team_obs_space = self.postprocessors[team].observation_space(raw_obs_space)
        self.flat_obs_space = pufferlib.new_emulation.flatten_space(team_obs_space)  
        return self.flat_obs_space

    def action_space(self, team):
        raw_atn_space = {agent: self.env.action_space(agent) for agent in team}
        team_atn_space = self.postprocessors[team].action_space(raw_atn_space)
        flat_space = pufferlib.new_emulation.flatten_space(raw_atn_space)
        return flat_space

    def reset(self):
        obs = self.env.reset()
        obs = pufferlib.new_emulation.group_into_teams(obs, self.teams)
        obs = pufferlib.new_emulation.flatten_space(obs)
        return obs

    def step(self, actions):
        actions = pufferlib.new_emulation.ungroup_from_teams(actions)
        actions = pufferlib.new_emulation.unpack_actions(actions)
        obs, rewards, dones, infos = self.env.step(actions)
        obs, rewards, dones = pufferlib.new_emulation.group_into_teams(
                self.teams, obs, rewards, dones)

        for team in obs:
            obs[team] = pufferlib.new_emulation.flatten_to_array(obs[team], self._flat_observation_space[team], np.float32)
 


class Policy(pufferlib.models.Policy):
  def __init__(self, binding, input_size=256, hidden_size=256, output_size=256):
      '''Simple custom PyTorch policy subclassing the pufferlib BasePolicy

      This requires only that you structure your network as an observation encoder,
      an action decoder, and a critic function. If you use our LSTM support, it will
      be added between the encoder and the decoder.
      '''
      super().__init__(binding)
      # :/
      self.raw_single_observation_space = binding.raw_single_observation_space

      # A dumb example encoder that applies a linear layer to agent self features
      observation_size = binding.raw_single_observation_space['Entity'].shape[1]

      self.embedding = torch.nn.Embedding(NUM_ATTRS*256, 32)
      self.tile_conv_1 = torch.nn.Conv2d(96, 32, 3)
      self.tile_conv_2 = torch.nn.Conv2d(32, 8, 3)
      self.tile_fc = torch.nn.Linear(8*11*11, input_size)

      self.entity_fc = torch.nn.Linear(23*32, input_size)

      self.proj_fc = torch.nn.Linear(2*input_size, input_size)

      self.decoders = torch.nn.ModuleList([torch.nn.Linear(hidden_size, n)
              for n in binding.single_action_space.nvec])
      self.value_head = torch.nn.Linear(hidden_size, 1)

  def critic(self, hidden):
      return self.value_head(hidden)

  def encode_observations(self, env_outputs):
    # TODO: Change 0 for teams when teams are added
    env_outputs = self.binding.unpack_batched_obs(env_outputs)[0]

    tile = env_outputs['Tile']
    # Center on player
    # This is cursed without clone??
    tile[:, :, :2] -= tile[:, 112:113, :2].clone() 
    tile[:, :, :2] += 7
    tile = self.embedding(
        tile.long().clip(0, 255) + tile_offset.to(tile.device)
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
    entity_ids = agentEmb[:,:,EntityId]
    mask = (entity_ids == my_id.unsqueeze(1)) & (entity_ids != 0)
    mask = mask.int()
    row_indices = torch.where(mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1)))
    entity = agentEmb[torch.arange(agentEmb.shape[0]), row_indices]

    entity = self.embedding(
        entity.long().clip(0, 255) + entity_offset.to(entity.device)
    )
    agents, attrs, embed = entity.shape
    entity = entity.view(agents, attrs*embed)

    entity = self.entity_fc(entity)
    entity = F.relu(entity)

    obs = torch.cat([tile, entity], dim=-1)
    return self.proj_fc(obs), None

  def decode_actions(self, hidden, lookup, concat=True):
      actions = [dec(hidden) for dec in self.decoders]
      if concat:
          return torch.cat(actions, dim=-1)
      return actions