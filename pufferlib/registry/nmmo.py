from pdb import set_trace as T

import torch
import torch.nn.functional as F


import pufferlib
import pufferlib.emulation
import pufferlib.models

from nmmo.entity.entity import EntityState


NUM_ATTRS = 26
EntityId = EntityState.State.attr_name_to_col["id"]
tile_offset = torch.tensor([i*256 for i in range(3)])
agent_offset = torch.tensor([i*256 for i in range(3, 26)])

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

      self.agent_fc = torch.nn.Linear(23*32, hidden_size)
      self.my_agent_fc = torch.nn.Linear(23*32, input_size)

      self.proj_fc = torch.nn.Linear(2*input_size, input_size)
      self.attack_fc = torch.nn.Linear(hidden_size, hidden_size)

      action_dims = binding.single_action_space.nvec
      action_dims = [action_dims[0], *action_dims[2:]]
      self.decoders = torch.nn.ModuleList([torch.nn.Linear(hidden_size, n) for n in action_dims])
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
    agents = env_outputs["Entity"]
    my_id = env_outputs["AgentId"][:,0]
    agent_ids = agents[:,:,EntityId]
    mask = (agent_ids == my_id.unsqueeze(1)) & (agent_ids != 0)
    mask = mask.int()
    row_indices = torch.where(mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1)))

    agent_embeddings = self.embedding(
        agents.long().clip(0, 255) + agent_offset.to(agents.device)
    )
    batch, agent, attrs, embed = agent_embeddings.shape

    # Embed each feature separately
    agent_embeddings = agent_embeddings.view(batch, agent, attrs*embed)
    my_agent_embeddings = agent_embeddings[torch.arange(agents.shape[0]), row_indices]

    # Project to input of recurrent size
    agent_embeddings = self.agent_fc(agent_embeddings)
    my_agent_embeddings = self.my_agent_fc(my_agent_embeddings)
    my_agent_embeddings = F.relu(my_agent_embeddings)

    obs = torch.cat([tile, my_agent_embeddings], dim=-1)
    return self.proj_fc(obs), agent_embeddings

  def decode_actions(self, hidden, lookup, concat=True):
      attack_key = self.attack_fc(hidden)
      attack_logits = torch.matmul(lookup, attack_key.unsqueeze(-1)).squeeze(-1)

      actions = [dec(hidden) for dec in self.decoders]
      actions.insert(1, attack_logits)
      if concat:
          return torch.cat(actions, dim=-1)
      return actions