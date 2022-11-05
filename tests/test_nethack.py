from pdb import set_trace as trace

from collections import defaultdict

from typing import Dict, Tuple

import numpy as np
from numpy import ndarray
import gym
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F

import ray
from ray.air import CheckpointConfig
from ray.air.config import RunConfig
from ray.air.config import ScalingConfig  
from ray.tune.registry import register_env
from ray.tune.tuner import Tuner
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.train.rl.rl_trainer import RLTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import pufferlib

import nle
from nle import nethack

def _step_to_range(delta, num_steps):
    """Range of `num_steps` integers with distance `delta` centered around zero."""
    return delta * torch.arange(-num_steps // 2, num_steps // 2)

def make_policy(
        config,
        observation_shape,
        num_actions,
        use_lstm,
        embedding_dim=32,
        crop_dim=9,
        num_layers=5):
    class Policy(RecurrentNetwork, nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            nn.Module.__init__(self)
            self.config = config

            self.value  = nn.Linear(config.HIDDEN, 1)
            self.lstm = pufferlib.torch.BatchFirstLSTM(config.HIDDEN, config.HIDDEN)

        def get_initial_state(self):
            return [self.value.weight.new(1, self.config.HIDDEN).zero_(),
                    self.value.weight.new(1, self.config.HIDDEN).zero_()]

        def forward_rnn(self, x, state, seq_lens):
            B, TT, _  = x.shape
            x         = x.reshape(B*TT, -1)

            lookup    = self.input(x)
            hidden, _ = self.policy(lookup)

            hidden        = hidden.view(B, TT, self.config.HIDDEN)
            hidden, state = self.lstm(hidden, state)
            hidden        = hidden.reshape(B*TT, self.config.HIDDEN)

            self.val = self.value(hidden).squeeze(-1)
            logits   = self.output(hidden, lookup)

            flat_logits = []
            for atn in nmmo.Action.edges(self.config):
                for arg in atn.edges:
                    flat_logits.append(logits[atn][arg])

            flat_logits = torch.cat(flat_logits, 1)
            return flat_logits, state

        def value_function(self):
            return self.val.view(-1)

    class Crop(nn.Module):
        """Helper class for NetHackNet below."""

        def __init__(self, height, width, height_target, width_target):
            super(Crop, self).__init__()
            self.width = width
            self.height = height
            self.width_target = width_target
            self.height_target = height_target
            width_grid = _step_to_range(2 / (self.width - 1), self.width_target)[
                None, :
            ].expand(self.height_target, -1)
            height_grid = _step_to_range(2 / (self.height - 1), height_target)[
                :, None
            ].expand(-1, self.width_target)

            # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
            self.register_buffer("width_grid", width_grid.clone())
            self.register_buffer("height_grid", height_grid.clone())

        def forward(self, inputs, coordinates):
            """Calculates centered crop around given x,y coordinates.
            Args:
            inputs [B x H x W]
            coordinates [B x 2] x,y coordinates
            Returns:
            [B x H' x W'] inputs cropped and centered around x,y coordinates.
            """
            assert inputs.shape[1] == self.height
            assert inputs.shape[2] == self.width

            inputs = inputs[:, None, :, :].float()

            x = coordinates[:, 0]
            y = coordinates[:, 1]

            x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
            y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

            grid = torch.stack(
                [
                    self.width_grid[None, :, :] + x_shift[:, None, None],
                    self.height_grid[None, :, :] + y_shift[:, None, None],
                ],
                dim=3,
            )

            # TODO: only cast to int if original tensor was int
            return (
                torch.round(F.grid_sample(inputs, grid, align_corners=True))
                .squeeze(1)
                .long()
            )


    class NetHackNet(RecurrentNetwork, nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            nn.Module.__init__(self)

            self.glyph_shape = observation_shape["glyphs"].shape
            self.blstats_size = observation_shape["blstats"].shape[0]

            self.num_actions = num_actions
            self.use_lstm = use_lstm

            self.H = self.glyph_shape[0]
            self.W = self.glyph_shape[1]

            self.k_dim = embedding_dim
            self.h_dim = 512

            self.crop_dim = crop_dim

            self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

            self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

            K = embedding_dim  # number of input filters
            F = 3  # filter dimensions
            S = 1  # stride
            P = 1  # padding
            M = 16  # number of intermediate filters
            Y = 8  # number of output filters
            L = num_layers  # number of convnet layers

            in_channels = [K] + [M] * (L - 1)
            out_channels = [M] * (L - 1) + [Y]

            def interleave(xs, ys):
                return [val for pair in zip(xs, ys) for val in pair]

            conv_extract = [
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=(F, F),
                    stride=S,
                    padding=P,
                )
                for i in range(L)
            ]

            self.extract_representation = nn.Sequential(
                *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
            )

            # CNN crop model.
            conv_extract_crop = [
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=(F, F),
                    stride=S,
                    padding=P,
                )
                for i in range(L)
            ]

            self.extract_crop_representation = nn.Sequential(
                *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
            )

            out_dim = self.k_dim
            # CNN over full glyph map
            out_dim += self.H * self.W * Y

            # CNN crop model.
            out_dim += self.crop_dim**2 * Y

            self.embed_blstats = nn.Sequential(
                nn.Linear(self.blstats_size, self.k_dim),
                nn.ReLU(),
                nn.Linear(self.k_dim, self.k_dim),
                nn.ReLU(),
            )

            self.fc = nn.Sequential(
                nn.Linear(out_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU(),
            )

            if self.use_lstm:
                self.core = pufferlib.torch.BatchFirstLSTM(self.h_dim, self.h_dim, num_layers=1)

            self.policy = nn.Linear(self.h_dim, self.num_actions)
            self.baseline = nn.Linear(self.h_dim, 1)

        def get_initial_state(self, batch_size=1):
            if not self.use_lstm:
                return tuple()
            return tuple(
                torch.zeros(self.core.num_layers, self.core.hidden_size)
                for _ in range(2)
            )

        def value_function(self):
            return self.value.view(-1)

        def _select(self, embed, x):
            # Work around slow backward pass of nn.Embedding, see
            # https://github.com/pytorch/pytorch/issues/24912
            out = embed.weight.index_select(0, x.reshape(-1))
            return out.reshape(x.shape + (-1,))

        def forward_rnn(self, env_outputs, core_state, seq_lens):
            B, T, _ = env_outputs.shape
            env_outputs = env_outputs.reshape(B*T, -1)
            env_outputs = pufferlib.emulation.unpack_batched_obs(
                    observation_shape, env_outputs)

            # -- [T x B x H x W]
            glyphs = env_outputs["glyphs"]

            # -- [T x B x F]
            blstats = env_outputs["blstats"]

            #T, B, *_ = glyphs.shape

            # -- [B' x H x W]
            #glyphs = torch.flatten(glyphs, 0, 1)  # Merge time and batch.

            # -- [B' x F]
            #blstats = blstats.view(T * B, -1).float()

            # -- [B x H x W]
            glyphs = glyphs.long()
            # -- [B x 2] x,y coordinates
            coordinates = blstats[:, :2]
            # TODO ???
            # coordinates[:, 0].add_(-1)

            # -- [B x F]
            blstats = blstats.view(T * B, -1).float()
            # -- [B x K]
            blstats_emb = self.embed_blstats(blstats)

            assert blstats_emb.shape[0] == T * B

            reps = [blstats_emb]

            # -- [B x H' x W']
            crop = self.crop(glyphs, coordinates)

            # print("crop", crop)
            # print("at_xy", glyphs[:, coordinates[:, 1].long(), coordinates[:, 0].long()])

            # -- [B x H' x W' x K]
            crop_emb = self._select(self.embed, crop)

            # CNN crop model.
            # -- [B x K x W' x H']
            crop_emb = crop_emb.transpose(1, 3)  # -- TODO: slow?
            # -- [B x W' x H' x K]
            crop_rep = self.extract_crop_representation(crop_emb)

            # -- [B x K']
            crop_rep = crop_rep.view(T * B, -1)
            assert crop_rep.shape[0] == T * B

            reps.append(crop_rep)

            # -- [B x H x W x K]
            glyphs_emb = self._select(self.embed, glyphs)
            # glyphs_emb = self.embed(glyphs)
            # -- [B x K x W x H]
            glyphs_emb = glyphs_emb.transpose(1, 3)  # -- TODO: slow?
            # -- [B x W x H x K]
            glyphs_rep = self.extract_representation(glyphs_emb)

            # -- [B x K']
            glyphs_rep = glyphs_rep.view(T * B, -1)

            assert glyphs_rep.shape[0] == T * B

            # -- [B x K'']
            reps.append(glyphs_rep)

            st = torch.cat(reps, dim=1)

            # -- [B x K]
            st = self.fc(st)

            #hidden        = hidden.view(B, TT, self.config.HIDDEN)
            #hidden, state = self.lstm(hidden, state)
            #hidden        = hidden.reshape(B*TT, self.config.HIDDEN)

            if self.use_lstm:
                core_input = st.view(B, T, -1)
                core_output, core_state = self.core(core_input, core_state)
                core_output = core_output.reshape(B*T, self.core.hidden_size)

                '''
                core_output_list = []
                notdone = (~env_outputs["done"]).float()
                for input, nd in zip(core_input.unbind(), notdone.unbind()):
                    # Reset core state to zero whenever an episode ended.
                    # Make `done` broadcastable with (num_layers, B, hidden_size)
                    # states:
                    nd = nd.view(1, -1, 1)
                    core_state = tuple(nd * s for s in core_state)
                    output, core_state = self.core(input.unsqueeze(0), core_state)
                    core_output_list.append(output)
                core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
                '''
            else:
                core_output = st

            # -- [B x A]
            policy_logits = self.policy(core_output)
            # -- [B x A]
            baseline = self.baseline(core_output)
            self.value = baseline

            policy_logits = policy_logits.view(T, B, self.num_actions)
            return policy_logits, core_state

            if self.training:
                action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
            else:
                # Don't sample when testing.
                action = torch.argmax(policy_logits, dim=1)

            policy_logits = policy_logits.view(T, B, self.num_actions)
            baseline = baseline.view(T, B)
            action = action.view(T, B)

            return (
                dict(policy_logits=policy_logits, baseline=baseline, action=action),
                core_state,
            )

    return NetHackNet

class NMMOLogger(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        assert len(base_env.envs) == 1, 'One env per worker'
        env = base_env.envs[0].par_env

        inv_map = {agent.policyID: agent for agent in env.config.PLAYERS}

        stats = env.terminal()
        stats = {**stats['Player'], **stats['Env']}
        policy_ids = stats.pop('PolicyID')

        for key, vals in stats.items():
            policy_stat = defaultdict(list)

            # Per-population metrics
            for policy_id, v in zip(policy_ids, vals):
                policy_stat[policy_id].append(v)

            for policy_id, vals in policy_stat.items():
                policy = inv_map[policy_id].__name__

                k = f'{policy}_{policy_id}_{key}'
                episode.custom_metrics[k] = np.mean(vals)

        return super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            **kwargs
        )

class Config:
    HIDDEN = 32

def env_creator():
    import nle
    #return gym.make('NetHackScore-v0')
    env = nle.env.NLE
    env = pufferlib.emulation.SingleToMultiAgent(env)
    env = pufferlib.emulation.EnvWrapper(env)
    return env(Config())

# Dashboard fails on WSL
ray.init(include_dashboard=False, num_gpus=1)

#register_env('nethack', env_creator)
import nle
pufferlib.rllib.register_multiagent_env('nethack', env_creator)
#pufferlib.emulation.EnvWrapper(pufferlib.emulation.SingleToMultiAgent(nle.env.NLE)))

config = Config()
test_env = env_creator()
obs = test_env.reset()

orig_env = nle.env.NLE()
obs_space = orig_env.observation_space
num_actions = orig_env.action_space.n
 
ModelCatalog.register_custom_model('custom', 
        make_policy(config, obs_space, num_actions, use_lstm=True)) 

trainer = RLTrainer(
    scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
    algorithm="PPO",
    config={
        "num_gpus": 1,
        "num_workers": 4,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 16,
        "train_batch_size": 2**10,
        #"train_batch_size": 2**19,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 1,
        "framework": "torch",
        "env": "nethack",
        "model": {
            "custom_model": "custom",
            'custom_model_config': {'config': config},
            "max_seq_len": 16
        },
    }
)

tuner = Tuner(
    trainer,
    _tuner_kwargs={"checkpoint_at_end": True},
    run_config=RunConfig(
        local_dir='results',
        verbose=1,
        stop={"training_iteration": 5},
        checkpoint_config=CheckpointConfig(
            num_to_keep=5,
            checkpoint_frequency=1,
        ),
        callbacks=[
            #WandbLoggerCallback(
            #    project='NeuralMMO',
            #    api_key_file='wandb_api_key',
            #    log_config=False,
            #)
        ]
    ),
    param_space={
        #'callbacks': NMMOLogger,
    }
)

result = tuner.fit()[0]
print('Saved ', result.checkpoint)

#policy = RLCheckpoint.from_checkpoint(result.checkpoint).get_policy()