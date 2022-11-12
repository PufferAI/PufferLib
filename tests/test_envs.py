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

import pufferlib


class Policy(pufferlib.rllib.RecurrentNetwork):
    def __init__(self, *args, observation_space, action_space,
            input_size, hidden_size, num_lstm_layers, **kwargs):
        super().__init__(input_size, hidden_size, num_lstm_layers, *args, **kwargs)
        self.encoder = nn.Linear(observation_space.shape[0], hidden_size)
        self.decoders = nn.ModuleList([nn.Linear(hidden_size, n) for n in action_space.nvec])

    def encode_observations(self, env_outputs):
        return self.encoder(env_outputs), None

    def decode_actions(self, hidden, lookup):
        actions = [dec(hidden) for dec in self.decoders]
        return torch.cat(actions, dim=-1)


# Dashboard fails on WSL
ray.init(include_dashboard=False, num_gpus=1)
ModelCatalog.register_custom_model('custom', Policy)

import nle, nmmo
from pettingzoo.magent import battle_v3
from pettingzoo.butterfly import knights_archers_zombies_v8 as kaz
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from smac.env.pettingzoo.StarCraft2PZEnv import _parallel_env as smac_env

env_classes = {
    'nethack': (pufferlib.emulation.wrap(nle.env.NLE), []),
    'nmmo': (pufferlib.emulation.wrap(nmmo.Env), []),
    'kaz': (pufferlib.emulation.wrap(aec_to_parallel_wrapper), [kaz.raw_env()]),
    'magent': (pufferlib.emulation.wrap(aec_to_parallel_wrapper), [battle_v3.env()]),
    #'smac': (pufferlib.emulation.wrap(smac_env), [1000]),
}

for name, (env_cls, env_args) in env_classes.items():
    env_creator = lambda: env_cls(*env_args)
    test_env = env_creator()

    pufferlib.utils.check_env(test_env)
    pufferlib.rllib.register_env(name, env_creator)

    observation_space = test_env.observation_space(test_env.possible_agents[0])
    action_space = test_env.action_space(test_env.possible_agents[0])

    trainer = RLTrainer(
        scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
        algorithm="PPO",
        config={
            "num_gpus": 1,
            "num_workers": 4,
            "num_envs_per_worker": 1,
            "rollout_fragment_length": 16,
            "train_batch_size": 2**10,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 1,
            "framework": "torch",
            "env": name,
            "model": {
                "custom_model": "custom",
                'custom_model_config': {
                    'observation_space': observation_space,
                    'action_space': action_space,
                    'input_size': 32,
                    'hidden_size': 32,
                    'num_lstm_layers': 1,
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