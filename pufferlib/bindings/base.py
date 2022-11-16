from pdb import set_trace as T

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

from pufferlib.emulation import wrap
from pufferlib.frameworks import BasePolicy


def auto(env_cls, env_args=[], env_name=None):
    class AutoBound(Base):
        def __init__(self):
            super().__init__()

            self.env_cls = wrap(env_cls)
            self.env_args = env_args

            self.env_name = env_name
            if env_name is None:
                self.env_name = env_cls.__name__

            self.env_creator = lambda: self.env_cls(*self.env_args)
            self.test_env = self.env_creator()

    return AutoBound()

class Base:
    def __init__(self):
        self.env_name = 'base'
        self.env_cls = None
        self.env_args = []
        self.config = {}
        self.policy = Policy

    @property
    def custom_model_config(self):
        return {
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'input_size': 32,
            'hidden_size': 32,
            'lstm_layers': 0,
        }

    @property
    def observation_space(self):
        return self.test_env.observation_space(self.test_env.possible_agents[0])

    @property
    def action_space(self):
        return self.test_env.action_space(self.test_env.possible_agents[0])


class Policy(BasePolicy):
    def __init__(self, observation_space, action_space,
            input_size, hidden_size, lstm_layers):
        super().__init__(input_size, hidden_size, lstm_layers)
        self.encoder = nn.Linear(observation_space.shape[0], hidden_size)
        self.decoders = nn.ModuleList([nn.Linear(hidden_size, n)
                for n in action_space.nvec])
        self.value_head = nn.Linear(hidden_size, 1)

    def critic(self, hidden):
        return self.value_head(hidden)

    def encode_observations(self, env_outputs):
        return self.encoder(env_outputs), None

    def decode_actions(self, hidden, lookup):
        actions = [dec(hidden) for dec in self.decoders]
        return torch.cat(actions, dim=-1)