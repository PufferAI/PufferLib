from pdb import set_trace as T

from collections import defaultdict

from typing import Dict, Tuple

import inspect
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

from pufferlib.emulation import PufferWrapper
from pufferlib.frameworks import BasePolicy
from pufferlib.utils import is_multiagent


def auto(env_name=None, env=None, env_cls=None, env_args=[], env_kwargs={}, **kwargs):
    class AutoBound(Base):
        def __init__(self):
            if env_cls is not None:
                #assert inspect.isclass(env_cls)
                self.env_cls = PufferWrapper(env_cls, **kwargs)
                self.env_name = env_cls.__name__
            elif env is not None:
                assert not inspect.isclass(env)
                self.env_cls = PufferWrapper(env, **kwargs)
                self.env_name = env.__class__.__name__
            else:
                raise Exception('Specify env or env_cls')

            if env_name is not None:
                self.env_name = env_name

            super().__init__(self.env_name, self.env_cls, env_args, env_kwargs)
            self.policy = Policy

    return AutoBound()

class Base:
    def __init__(self, env_name, env_cls, env_args=[], env_kwargs={}):
        self.env_name = env_name
        self.env_cls = env_cls

        self.env_args = env_args
        self.env_kwargs = env_kwargs

        self.env_creator = lambda: self.env_cls(*self.env_args, **self.env_kwargs)

        local_env = self.env_creator()
        self.default_agent = local_env.possible_agents[0]
        self.single_observation_space = local_env.observation_space(self.default_agent)
        self.single_action_space = local_env.action_space(self.default_agent)


    '''
    @property
    def single_observation_space(self):
        return self.local_env.observation_space(
                self.local_env.possible_agents[0])

    @property
    def single_action_space(self):
        return self.local_env.action_space(
                self.local_env.possible_agents[0])

    '''
    #def env_creator(self):
    #    return self.env_cls(*self.env_args, **self.env_kwargs)

    @property
    def custom_model_config(self):
        return {
            'observation_space': self.single_observation_space,
            'action_space': self.single_action_space,
            'input_size': 32,
            'hidden_size': 32,
            'lstm_layers': 0,
        }


class Policy(BasePolicy):
    def __init__(self, observation_space, action_space,
            input_size, hidden_size, lstm_layers):
        super().__init__(input_size, hidden_size, lstm_layers)
        self.observation_space = observation_space
        self.action_space = action_space
        self.encoder = nn.Linear(observation_space.shape[0], hidden_size)
        self.decoders = nn.ModuleList([nn.Linear(hidden_size, n)
                for n in action_space.nvec])
        self.value_head = nn.Linear(hidden_size, 1)

    def critic(self, hidden):
        return self.value_head(hidden)

    def encode_observations(self, env_outputs):
        return self.encoder(env_outputs), None

    def decode_actions(self, hidden, lookup, concat=True):
        actions = [dec(hidden) for dec in self.decoders]
        if concat:
            return torch.cat(actions, dim=-1)
        return actions