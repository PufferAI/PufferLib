from pdb import set_trace as T
import numpy as np

import os
import inspect

from ray.train.rl import RLCheckpoint
from ray.train.rl.rl_predictor import RLPredictor as RLlibPredictor
from ray.tune.registry import register_env as tune_register_env
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as RLLibRecurrentNetwork
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env import ParallelPettingZooEnv

import pettingzoo

import torch

import pufferlib


def register_env(name, env_creator):
    assert type(name) == str, 'Name must be a str'
    tune_register_env(name, lambda config: ParallelPettingZooEnv(env_creator())) 

def read_checkpoints(tune_path):
     folders = sorted([f.path for f in os.scandir(tune_path) if f.is_dir()])
     assert len(folders) <= 1, 'Tune folder contains multiple trials'

     if len(folders) == 0:
        return []

     all_checkpoints = []
     trial_path = folders[0]

     for f in os.listdir(trial_path):
        if not f.startswith('checkpoint'):
            continue

        checkpoint_path = os.path.join(trial_path, f)
        all_checkpoints.append([f, RLCheckpoint(checkpoint_path)])

     return all_checkpoints

def create_policies(n):
    return {f'policy_{i}': 
        PolicySpec(
            policy_class=None,
            observation_space=None,
            action_space=None,
            config={"gamma": -1.85},
        )
        for i in range(n)
    }

class RecurrentNetwork(RLLibRecurrentNetwork, torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.nn.Module.__init__(self)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.value_head = torch.nn.Linear(hidden_size, 1)
        self.lstm = pufferlib.torch.BatchFirstLSTM(input_size, hidden_size, num_layers)

    def get_initial_state(self, batch_size=1):
        return tuple(
            torch.zeros(self.lstm.num_layers, self.lstm.hidden_size)
            for _ in range(2)
        )

    def value_function(self):
        return self.value.view(-1)

    def encode_observations(self, flat_observations):
        return hidden, lookup

    def decode_actions(self, flat_hidden, lookup):
        return logits

    def forward_rnn(self, x, state, seq_lens):
        B, TT, _ = x.shape
        x = x.reshape(B*TT, -1)

        hidden, lookup = self.encode_observations(x)
        assert hidden.shape == (B*TT, self.hidden_size)

        hidden = hidden.view(B, TT, self.hidden_size)
        hidden, state = self.lstm(hidden, state)
        hidden = hidden.reshape(B*TT, self.hidden_size)

        self.value = self.value_head(hidden)
        logits = self.decode_actions(hidden, lookup)

        return logits, state

class RLPredictor(RLlibPredictor):
    def predict(self, data, **kwargs):
        batch = data.shape[0]
        #data = data.reshape(batch, -1)
        data = data.squeeze()
        result = super().predict(data, **kwargs)
        if type(result) == dict:
            result = np.stack(list(result.values()), axis=-1)
        return result
        result = np.concatenate(list(result.values())).reshape(1, -1)
        return result

class Callbacks(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, trainer, **kwargs) -> None:
        '''Run after 1 epoch at the trainer level'''
        return super().on_train_result(
            algorithm=algorithm,
            result=result,
            trainer=trainer,
            **kwargs
        )

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        self._on_episode_end(worker, base_env, policies, episode, **kwargs)
        return super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            **kwargs
        )