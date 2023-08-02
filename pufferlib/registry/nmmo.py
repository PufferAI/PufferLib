from pdb import set_trace as T

import gym

import numpy as np
import torch
import torch.nn.functional as F


import pufferlib
import pufferlib.emulation
import pufferlib.models
import pufferlib.exceptions

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

        pufferlib.new_emulation.check_teams(self.env, teams)
        self.postprocessors = {
            team: postprocessor_cls(self.env, self.teams, team)
            for team in teams
        }

        self.possible_agents = self.env.possible_agents

        team = list(teams.keys())[0]
        self.observation_space(team)
        self.action_space(team)

        self._step = 0

    def observation_space(self, team):
        '''Returns the observation space for a single agent'''
        if team not in self.teams:
            raise pufferlib.exceptions.InvalidAgentError(team, self._teams)


        raw_obs_space = {agent: self.env.observation_space(agent) for agent in self.teams[team]}

        raw_obs = {agent: space.sample() for agent, space in raw_obs_space.items()}
        team_obs = self.postprocessors[team].features(raw_obs, 0)

        self.featurized_obs_space = pufferlib.new_emulation.make_space_like(team_obs)
        self.flat_obs_space = pufferlib.new_emulation.flatten_space(self.featurized_obs_space)  
        flat_obs = pufferlib.new_emulation.flatten_to_array(team_obs, self.flat_obs_space)

        mmin, mmax = pufferlib.utils._get_dtype_bounds(flat_obs.dtype)
        self.pad_obs = 0 * flat_obs
        self.box_obs_space = gym.spaces.Box(
            low=mmin, high=mmax,
            shape=flat_obs.shape, dtype=flat_obs.dtype
        )

        return self.box_obs_space

    def action_space(self, team):
        if team not in self.teams:
            raise pufferlib.exceptions.InvalidAgentError(team, self._teams)

        raw_atn_space = {agent: self.env.action_space(agent) for agent in self.teams[team]}
        self.flat_action_space = pufferlib.new_emulation.flatten_space(raw_atn_space)

        multidiscrete_space = pufferlib.new_emulation.convert_to_multidiscrete(self.flat_action_space)
        return multidiscrete_space

    def reset(self):
        obs = self.env.reset()
        team_obs = pufferlib.new_emulation.group_into_teams(self.teams, obs)

        for team in self.teams:
            self.postprocessors[team].reset(team_obs[team])
            team_obs[team] = self.postprocessors[team].features(team_obs[team], team)
            team_obs[team] = pufferlib.new_emulation.flatten_to_array(
                team_obs[team], self.flat_obs_space)

        self.pad_obs = team_obs[list(team_obs.keys())[0]]
        return team_obs

    def step(self, actions):
        for team in self.teams:
            actions[team] = self.postprocessors[team].actions(actions[team], self._step)

        pufferlib.new_emulation.check_spaces(actions, self.action_space)
        actions = pufferlib.new_emulation.unpack_actions(actions, self.teams, self.flat_action_space)
        actions = pufferlib.new_emulation.ungroup_from_teams(actions)
        obs, rewards, dones, infos = self.env.step(actions)

        team_obs, rewards, dones = pufferlib.new_emulation.group_into_teams(
                self.teams, obs, rewards, dones)

        for team in self.teams:
            team_obs[team] = self.postprocessors[team].features(team_obs[team], team)
            team_obs[team] = pufferlib.new_emulation.flatten_to_array(
                team_obs[team], self.flat_obs_space)

        obs = team_obs
        obs = pufferlib.new_emulation.pad_agent_data(obs, self.teams, self.pad_obs)
        rewards = pufferlib.new_emulation.pad_agent_data(rewards, self.teams, 0)
        dones = pufferlib.new_emulation.pad_agent_data(dones, self.teams, False)
        infos = pufferlib.new_emulation.pad_agent_data(infos, self.teams, {})

        for team in self.teams:
            env_done = team_done = False
            rewards[team] = 0
            infos[team] = self.postprocessors[team].infos(rewards[team], env_done, team_done, infos[team], self._step)
        
        pufferlib.new_emulation.check_spaces(obs, self.observation_space)
        return obs, rewards, dones, infos
 


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