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

class Policy(pufferlib.rllib.RecurrentNetwork):
    def __init__(self, *args,
            observation_shape, num_actions, use_lstm,
            embedding_dim, crop_dim, num_layers,
            input_size, hidden_size, num_lstm_layers,
            **kwargs):
        super().__init__(input_size, hidden_size, num_lstm_layers, *args, **kwargs)

        self.observation_shape = observation_shape
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

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def encode_observations(self, env_outputs):
        TB, _ = env_outputs.shape
        env_outputs = pufferlib.emulation.unpack_batched_obs(
                self.observation_shape, env_outputs)

        glyphs = env_outputs["glyphs"].long()
        blstats = env_outputs["blstats"]

        # BL Stats
        coordinates = blstats[:, :2]
        blstats_emb = self.embed_blstats(blstats)
        assert blstats_emb.shape[0] == TB

        # Crop
        crop = self.crop(glyphs, coordinates)
        crop_emb = self._select(self.embed, crop).transpose(1, 3)
        crop_rep = self.extract_crop_representation(crop_emb).view(TB, -1)
        assert crop_rep.shape[0] == TB

        # Glyphs
        glyphs_emb = self._select(self.embed, glyphs).transpose(1, 3)
        glyphs_rep = self.extract_representation(glyphs_emb).view(TB, -1)
        assert glyphs_rep.shape[0] == TB

        st = torch.cat([blstats_emb, crop_rep, glyphs_rep], dim=1)
        st = self.fc(st)

        return st, None

    def decode_actions(self, hidden, lookup):
        return self.policy(hidden)


# Dashboard fails on WSL
ray.init(include_dashboard=False, num_gpus=1)

orig_env = nle.env.NLE()
env_cls = pufferlib.emulation.wrap(nle.env.NLE,
        emulate_flat_atn=False)
env_creator = lambda: env_cls()

pufferlib.rllib.register_env('nethack', env_cls)
ModelCatalog.register_custom_model('custom', Policy)

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
            'custom_model_config': {
                'embedding_dim': 32,
                'crop_dim': 9,
                'num_layers': 5,
                'input_size': 512,
                'hidden_size': 512,
                'num_lstm_layers': 1,
                'observation_shape': orig_env.observation_space,
                'num_actions': orig_env.action_space.n,
                'use_lstm': True,
            },
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
        ]
    ),
    param_space={
    }
)

result = tuner.fit()[0]
print('Saved ', result.checkpoint)