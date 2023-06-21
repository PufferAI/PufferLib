import torch
from torch.nn import functional as F

from nmmo.entity.entity import EntityState
EntityId = EntityState.State.attr_name_to_col["id"]

import pufferlib
import pufferlib.vectorization
from clean_pufferl import CleanPuffeRL


class Agent(pufferlib.models.Policy):
    def __init__(self, binding, input_size=128, hidden_size=256):
        '''Simple custom PyTorch policy subclassing the pufferlib BasePolicy

        This requires only that you structure your network as an observation encoder,
        an action decoder, and a critic function. If you use our LSTM support, it will
        be added between the encoder and the decoder.
        '''
        super().__init__(binding)
        self.raw_single_observation_space = binding.raw_single_observation_space

        # A dumb example encoder that applies a linear layer to agent self features
        observation_size = binding.raw_single_observation_space['Entity'].shape[1]

        self.tile_conv_1 = torch.nn.Conv2d(3, 32, 3)
        self.tile_conv_2 = torch.nn.Conv2d(32, 8, 3)
        self.tile_fc = torch.nn.Linear(8*11*11, input_size)

        self.entity_fc = torch.nn.Linear(23, input_size)

        self.proj_fc = torch.nn.Linear(256, input_size)

        self.decoders = torch.nn.ModuleList([torch.nn.Linear(hidden_size, n)
                for n in binding.single_action_space.nvec])
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def critic(self, hidden):
        return self.value_head(hidden)

    def encode_observations(self, env_outputs):
        # TODO: Change 0 for teams when teams are added
        env_outputs = self.binding.unpack_batched_obs(env_outputs)[0]

        tile = env_outputs['Tile']
        agents, tiles, features = tile.shape
        tile = tile.transpose(1, 2).view(agents, features, 15, 15)

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

        #entity = env_outputs['Entity'][:, 0, :]
        entity = self.entity_fc(entity)
        entity = F.relu(entity)

        obs = torch.cat([tile, entity], dim=-1)
        return self.proj_fc(obs), None

    def decode_actions(self, hidden, lookup, concat=True):
        actions = [dec(hidden) for dec in self.decoders]
        if concat:
            return torch.cat(actions, dim=-1)
        return actions

from pufferlib.registry import nmmo
device = 'cuda'

import nmmo
binding = pufferlib.emulation.Binding(
        env_cls=nmmo.Env,
        env_name='Neural MMO',
        emulate_const_horizon=1024,
    )

agent = pufferlib.frameworks.cleanrl.make_policy(
        Agent, recurrent_args=[128, 128],
        recurrent_kwargs={'num_layers': 1}
    )(binding, 128, 128).to(device)

trainer = CleanPuffeRL(binding, agent,
        num_buffers=2, num_envs=8, num_cores=4,
        batch_size=2**14,
        vec_backend=pufferlib.vectorization.multiprocessing.VecEnv)

#trainer = CleanPuffeRL(binding, agent,
#        num_buffers=1, num_envs=1, num_cores=1,
#        batch_size=2**14,
#        vec_backend=pufferlib.vectorization.serial.VecEnv)

#trainer.load_model(path)
trainer.init_wandb()

data = trainer.allocate_storage()

num_updates = 10000
for update in range(trainer.update+1, num_updates + 1):
    trainer.evaluate(agent, data)
    trainer.train(agent, data, batch_rows=1024)

trainer.close()