'''RLlib support under construction as we focus on stable CleanRL support for v0.2

Still supported via Discord, but not officially stable
'''

from pdb import set_trace as T
import numpy as np
import torch

import os

from ray.train.rl.rl_predictor import RLPredictor as RLlibPredictor
from ray.train.rl import RLCheckpoint
from ray.tune.registry import register_env as tune_register_env
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as RLLibRecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env import ParallelPettingZooEnv

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

def make_policy(policy_cls, lstm_layers):
    '''Wrap a PyTorch model for use with RLLib 

    Args:
        policy_cls: A pufferlib.models.Policy subclass that implements the PufferLib model API
        lstm_layers: The number of LSTM layers to use. If 0, no LSTM is used

    Returns:
        A new RLlib model class wrapping your model
    '''
    assert issubclass(policy_cls, pufferlib.binding.Policy)

    if lstm_layers > 0:
        policy_cls = pufferlib.frameworks.base.make_recurrent_policy(policy_cls)

        class RLLibPolicy(RLLibRecurrentNetwork, policy_cls):
            def __init__(self, *args, **kwargs):
                policy_cls.__init__(self, **kwargs)
                RLLibRecurrentNetwork.__init__(self, *args)

            def get_initial_state(self, batch_size=1):
                return tuple(
                    torch.zeros(self.lstm.num_layers, self.lstm.hidden_size)
                    for _ in range(2)
                )

            def value_function(self):
                return self.value.view(-1)

            def forward_rnn(self, x, state, seq_lens):
                hidden, state, lookup = self.encode_observations(x, state)
                self.value = self.critic(hidden)
                logits = self.decode_actions(hidden, lookup)
                return logits, state

        return RLLibPolicy
    else:
        class RLlibPolicy(TorchModelV2, policy_cls):
            def __init__(self, *args, **kwargs):
                policy_cls.__init__(self, **kwargs)
                TorchModelV2.__init__(self, *args)

            def value_function(self):
                return self.value.view(-1)

            def forward(self, x, state, seq_lens):
                hidden, lookup = self.encode_observations(x['obs'].float())
                self.value = self.critic(hidden)
                logits = self.decode_actions(hidden, lookup)
                return logits, state

        return RLlibPolicy

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
