### PufferLib demo environment0
# Package parameters override defaults.
# Parameters for specific envs override packages
# You cannot specify any deeper than that.
from pdb import set_trace as T

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiDiscrete

import pufferlib
from pufferlib.models import LSTMWrapper

def setup_obs(sim):
    N = sim.reset_tensor().to_torch().shape[0]

    prep_counter = sim.prep_counter_tensor().to_torch()[0:N * 5, ...]
    agent_type = sim.agent_type_tensor().to_torch()[0:N * 5, ...]
    agent_data = sim.agent_data_tensor().to_torch()[0:N * 5, ...]
    box_data = sim.box_data_tensor().to_torch()[0:N * 5, ...]
    ramp_data = sim.ramp_data_tensor().to_torch()[0:N * 5, ...]
    visible_agents_mask = sim.visible_agents_mask_tensor().to_torch()[0:N * 5, ...]
    visible_boxes_mask = sim.visible_boxes_mask_tensor().to_torch()[0:N * 5, ...]
    visible_ramps_mask = sim.visible_ramps_mask_tensor().to_torch()[0:N * 5, ...]
    lidar_tensor = sim.lidar_tensor().to_torch()[0:N * 5, ...]

    # Add in an agent ID tensor
    id_tensor = torch.arange(5).float()

    id_tensor = id_tensor.to(device=prep_counter.device)
    id_tensor = id_tensor.view(1, 5).expand(prep_counter.shape[0] // 5, 5).reshape(
        prep_counter.shape[0], 1)

    obs_tensors = [
        prep_counter,
        agent_type,
        agent_data,
        lidar_tensor,
        id_tensor,
    ]

    num_obs_features = 0
    for tensor in obs_tensors:
        num_obs_features += math.prod(tensor.shape[1:])

    ent_tensors = [
        box_data,
        ramp_data,
    ]

    num_ent_features = 0
    for tensor in ent_tensors:
        num_ent_features += math.prod(tensor.shape[1:])

    obs_tensors += ent_tensors

    obs_tensors += [
        visible_agents_mask,
        visible_boxes_mask,
        visible_ramps_mask,
    ]

    return obs_tensors, num_obs_features, num_ent_features

def flatten(tensor):
    return tensor.view(tensor.shape[0], -1)


def process_obs(prep_counter,
                self_agent_data,
                other_agent_data,
                lidar,
                id_tensor,
                relative_box_obs,
                relative_ramp_obs,
                agent_type_mask,
                visible_agents_mask,
                visible_boxes_mask,
                visible_ramps_mask,
                task_embedding,
            ):
    # assert(not torch.isnan(prep_counter).any())
    # assert(not torch.isinf(prep_counter).any())

    # assert(not torch.isnan(agent_type_mask).any())
    # assert(not torch.isinf(agent_type_mask).any())

    # assert(not torch.isnan(agent_data).any())
    # assert(not torch.isinf(agent_data).any())

    # assert(not torch.isnan(relative_box_obs).any())
    # assert(not torch.isinf(relative_box_obs).any())

    # assert(not torch.isnan(relative_ramp_obs).any())
    # assert(not torch.isinf(relative_ramp_obs).any())

    # assert(not torch.isnan(lidar).any())
    # assert(not torch.isinf(lidar).any())

    # assert(not torch.isnan(visible_agents_mask).any())
    # assert(not torch.isinf(visible_agents_mask).any())

    # assert(not torch.isnan(visible_boxes_mask).any())
    # assert(not torch.isinf(visible_boxes_mask).any())

    # assert(not torch.isnan(visible_ramps_mask).any())
    # assert(not torch.isinf(visible_ramps_mask).any())


    # T()
    common = torch.cat([
            flatten(prep_counter / 200),
            flatten(agent_type_mask),
            id_tensor,
            # task_embedding
        ], dim=1)

    # T()

    not_common = [
            lidar,
            self_agent_data,
            other_agent_data,
            relative_box_obs,
            relative_ramp_obs,
            visible_agents_mask,
            visible_boxes_mask,
            visible_ramps_mask,
            task_embedding
        ]

    return (common, not_common)


class Recurrent(LSTMWrapper):
    def __init__(self, env, policy, input_size=256, hidden_size=256, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)
        self._state = None

    def get_state(self, x):
        batch_size = x.shape[0]
        if self._state is None or self._state[0].shape[1] != batch_size:
            self._state = (
                torch.zeros(0, batch_size, 0).to(x.device),
                torch.zeros(0, batch_size, 0).to(x.device)
            )
        return self._state


# The attention-based architecture from hide and seek
class Policy(nn.Module):
    # num_obs_features, num_entity_features, 128, num_channels, 4
    def __init__(self, env, num_embed_channels=128, num_out_channels=256, num_heads=4):
        super().__init__()
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        base_env = env.env.env.env
        self.num_embed_channels = num_embed_channels
        self.num_out_channels = num_out_channels
        self.num_heads = num_heads
        obs_features = base_env.num_obs_features
        entity_features = base_env.num_ent_features

        self.lidar_conv = nn.Conv1d(in_channels=43, out_channels=43,
                                    kernel_size=3, padding=1, # groups=50,
                                    padding_mode='circular')

        # Initialize the embedding layer for self
        self.self_embed = nn.Sequential(
            # remove other agents from obs for this
            nn.Linear(59 + num_embed_channels, num_embed_channels),
            nn.LayerNorm(num_embed_channels)
        )

        # Initialize a single embedding layer for the entities
        # self.entity_embed = nn.Sequential(
        #     nn.Linear(entity_features, num_embed_channels),
        #     nn.LayerNorm(num_embed_channels)
        # )

        # embed other agents directly
        self.others_embed = nn.Sequential(
            nn.Linear(14, num_embed_channels),
            nn.LayerNorm(num_embed_channels)
        )

        # Initialize a single embedding layer for the entities
        self.box_entity_embed = nn.Sequential(
            nn.Linear(17, num_embed_channels),
            nn.LayerNorm(num_embed_channels)
        )

        self.ramp_entity_embed = nn.Sequential(
            nn.Linear(14, num_embed_channels),
            nn.LayerNorm(num_embed_channels)
        )

        self.task_embed = nn.Sequential(
            nn.Linear(1536, num_embed_channels),
            nn.LayerNorm(num_embed_channels)
        )

        # self.other_embed = nn.Sequential(
        #     nn.Linear(14 + 17 + 14, num_embed_channels),
        #     nn.LayerNorm(num_embed_channels)
        #  )

        # Attention and feed-forward layers
        self.multihead_attn = nn.MultiheadAttention(embed_dim=num_embed_channels, num_heads=num_heads)
        # nn.LayerNorm(num_embed_channels),
        # nn.LeakyReLU(),
        # nn.MultiheadAttention(embed_dim=num_embed_channels, num_heads=num_heads),

        self.multihead_attn2 = nn.Sequential(
            nn.MultiheadAttention(embed_dim=num_embed_channels, num_heads=num_heads),
            # nn.LayerNorm(num_embed_channels),
            # nn.LeakyReLU(),
            # nn.MultiheadAttention(embed_dim=num_embed_channels, num_heads=num_heads),
        )

        self.ff = nn.Sequential(
            nn.Linear(num_embed_channels, num_out_channels),
            nn.LayerNorm(num_out_channels),
            nn.LeakyReLU(),
            # nn.Linear(num_out_channels, num_out_channels),
            # nn.LayerNorm(num_out_channels),
            # nn.LeakyReLU(),
        )

        self.decoders = torch.nn.ModuleList([torch.nn.Linear(num_out_channels, n)
              for n in env.single_action_space.nvec])

        self.value_head = nn.Linear(num_out_channels, 1)

        # self.policy = PolicyHead(env.action_space)

    def encode_observations(self, env_outputs):
        observations = pufferlib.pytorch.nativize_tensor(env_outputs, self.dtype)
        #observations = pufferlib.emulation.unpack_batched_obs(
        #    env_outputs.float(), self.unflatten_context)

        prep_counter = observations['prep_counter']
        self_agent_data = observations['self_agent_data']
        other_agent_data = observations['other_agent_data']
        lidar = observations['lidar']
        id_tensor = observations['id_tensor']
        relative_box_obs = observations['relative_box_obs']
        relative_ramp_obs = observations['relative_ramp_obs']
        agent_type_mask = observations['agent_type_mask']
        visible_agents_mask = observations['visible_agents_mask']
        visible_boxes_mask = observations['visible_boxes_mask']
        visible_ramps_mask = observations['visible_ramps_mask']
        task_embedding = observations['task_embedding']

        # TODO: change into how puffer passes information
        common, not_common = process_obs(prep_counter, self_agent_data, other_agent_data, lidar, id_tensor,
            relative_box_obs, relative_ramp_obs, agent_type_mask, visible_agents_mask,
            visible_boxes_mask, visible_ramps_mask, task_embedding)

        (lidar, self_agent_data, other_agent_data, box_data, ramp_data, visible_agents_mask,
            visible_boxes_mask, visible_ramps_mask, task_embedding) = not_common
        self_agent_data = self_agent_data.contiguous()
        other_agent_data = other_agent_data.contiguous()
        box_data = box_data.contiguous()
        ramp_data = ramp_data.contiguous()
        visible_agents_mask = visible_agents_mask.contiguous()
        visible_boxes_mask = visible_boxes_mask.contiguous()
        visible_ramps_mask = visible_ramps_mask.contiguous()

        B = common.shape[0]
        N = self_agent_data.shape[1]
        _, NE_O, F_agents = other_agent_data.shape
        _, NE_B, F_box = box_data.shape
        _, NE_R, F_ramp = ramp_data.shape

        # inds = torch.arange(B)
        # spread_indices = torch.arange(B) % N
        # spread_indices = torch.cat([torch.arange(N) for _ in range((B // N) + 1)])[inds]
        # self_observables = agent_data[inds, spread_indices, :]

        # print(B, N, self_observables.shape)
        lidar_plus_agent_data = torch.hstack((lidar, self_agent_data))

        lidar_processed = self.lidar_conv(lidar_plus_agent_data.unsqueeze(2))

        # the batch dim in task embedding are all the same. So we can just take the first one
        embedded_task = F.leaky_relu(self.task_embed(task_embedding[0]))
        embedded_task = embedded_task.expand(B, -1)

        x_self = torch.cat([
            common,
            flatten(lidar_processed),
            flatten(embedded_task),
            flatten(self_agent_data),
                            ], dim=1)

        # T()
        x_self = x_self.unsqueeze(-2)
        embed_self = F.leaky_relu(self.self_embed(x_self))

        other_agents_embedding = F.leaky_relu(self.others_embed(other_agent_data.view(-1, F_agents)).view(B, NE_O, -1))
        box_embedding = F.leaky_relu(self.box_entity_embed(box_data.view(-1, F_box)).view(B, NE_B, -1))
        ramp_embedding = F.leaky_relu(self.ramp_entity_embed(ramp_data.view(-1, F_ramp)).view(B, NE_R, -1))

        masked_box_embedding = box_embedding * visible_boxes_mask
        masked_ramp_embedding = ramp_embedding * visible_ramps_mask
        masked_other_agent_embedding = other_agents_embedding * visible_agents_mask

        embedded_entities = torch.cat([embed_self, masked_other_agent_embedding,
                                       masked_box_embedding, masked_ramp_embedding], dim=-2)
        # print(embedded_entities.shape) = torch.Size([num_worlds * (seekers + hiders), 1 + NE_O + NE_B + NE_R, 128])

        # need the unsqueezes for the attention calculation
        attn_output, attn_output_weights = self.multihead_attn(embedded_entities, embedded_entities, embedded_entities)
        # T()
        # attn_output = F.leaky_relu(attn_output)
        # T()
        # attn_output, attn_output_weights = self.multihead_attn2(attn_output, attn_output, attn_output)
        attn_output = attn_output.mean(dim=-2)

        # Feedforward network
        ff_out = self.ff(attn_output)

        return ff_out, None

    def decode_actions(self, hidden, lookup, concat=True):
        actions = [dec(hidden) for dec in self.decoders]
        value = self.value_head(hidden)
        return actions, value


if __name__ == '__main__':

    import gpu_hideseek
    import torch
    import numpy as np
    import sys
    import time
    import PIL
    import PIL.Image
    torch.manual_seed(0)
    import random
    random.seed(0)

    from environment import MadronaHideAndSeekWrapper, MadronaHideAndSeekWrapperSplitTaskGraph
    import numpy as np

    num_worlds = 25
    num_steps = 2500
    entities_per_world = 0
    reset_chance = 0.

    device = torch.device('cpu')


    sim = gpu_hideseek.HideAndSeekSimulator(
                        exec_mode = gpu_hideseek.madrona.ExecMode.CPU,
                        gpu_id = 0,
                        num_worlds = num_worlds,
                        sim_flags = gpu_hideseek.SimFlags.Default,
                        rand_seed = 10,
                        min_hiders = 3,
                        max_hiders = 3,
                        min_seekers = 2,
                        max_seekers = 2,
                        num_pbt_policies = 0,
        )
    sim.init()

    env = MadronaHideAndSeekWrapperSplitTaskGraph(sim)
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    obs, _ = env.reset()
    num_obs_features = env.num_obs_features
    num_entity_features = env.num_ent_features
    # obs_tensors, num_obs_features, num_entity_features = setup_obs(sim)

    base_network = EntitySelfAttentionPolicy(env).to(device)
    network = Recurrent(env, base_network).to(device)

    x = process_obs(**obs)
    state = network.get_state(x[0])

    print(network)

    logits, value, state = network.forward(x, state)

    multi_categorical = [torch.distributions.Categorical(logits=l) for l in logits]

    action = torch.stack([c.sample() for c in multi_categorical])
    logprob = torch.stack([c.log_prob(a) for c, a in zip(multi_categorical, action)]).T.sum(1)
    entropy = torch.stack([c.entropy() for c in multi_categorical]).T.sum(1)

    actions = action.T

    move_amount = actions[:, 0]
    move_angle = actions[:, 1]
    turn = actions[:, 2]
    grab = actions[:, 3]
    lock = actions[:, 4]
    # move_amount, move_angle, turn, grab, lock, value

    action = {'move_amount': move_amount, 'move_angle': move_angle,
              'turn': turn, 'grab': grab, 'lock': lock}
    ns, rew, done, tr, i = env.step(action)
    print(rew)
    _ = 0
