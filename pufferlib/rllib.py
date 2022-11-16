from pdb import set_trace as T
import numpy as np

from abc import ABC, abstractmethod

import os
import inspect
from copy import deepcopy

import torch

import ray

from ray.air import CheckpointConfig
from ray.air.config import RunConfig, ScalingConfig
from ray.tune.tuner import Tuner
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.train.rl import RLCheckpoint
from ray.train.rl.rl_trainer import RLTrainer
from ray.train.rl.rl_predictor import RLPredictor as RLlibPredictor
from ray.tune.registry import register_env as tune_register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as RLLibRecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env import ParallelPettingZooEnv

import pufferlib


def make_rllib_tuner(binding, *,
        algorithm='PPO',
        num_gpus=1,
        num_workers=4,
        num_envs_per_worker=1,
        rollout_fragment_length=16,
        train_batch_size=2**10,
        sgd_minibatch_size=128,
        num_sgd_iter=1,
        max_seq_len=16,
        training_steps=3,
        checkpoints_to_keep=5,
        checkpoint_frequency=1,):
    '''Provides sane defaults for RLlib'''

    ray.init(
        include_dashboard=False, # WSL Compatibility
        ignore_reinit_error=True,
        num_gpus=num_gpus,
    )

    env_cls = binding.env_cls
    env_args = binding.env_args
    name = binding.env_name

    policy = make_rllib_policy(binding.policy,
            lstm_layers=binding.custom_model_config['lstm_layers'])
    ModelCatalog.register_custom_model(name, policy)
    env_creator = lambda: env_cls(*env_args)
    test_env = env_creator()

    pufferlib.utils.check_env(test_env)
    pufferlib.rllib.register_env(name, env_creator)

    trainer = RLTrainer(
        algorithm=algorithm,
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=num_gpus>0
        ),
        config={
            "num_gpus": num_gpus,
            "num_workers": num_workers,
            "num_envs_per_worker": num_envs_per_worker,
            "rollout_fragment_length": rollout_fragment_length,
            "train_batch_size": train_batch_size,
            "sgd_minibatch_size": sgd_minibatch_size,
            "num_sgd_iter": num_sgd_iter,
            "framework": 'torch',
            "env": name,
            "model": {
                "custom_model": name,
                'custom_model_config': binding.custom_model_config,
                "max_seq_len": max_seq_len,
            },
        }
    )

    tuner = Tuner(
        trainer,
        _tuner_kwargs={"checkpoint_at_end": True},
        run_config=RunConfig(
            local_dir='results',
            verbose=1,
            stop={
                "training_iteration": training_steps
            },
            checkpoint_config=CheckpointConfig(
                num_to_keep=checkpoints_to_keep,
                checkpoint_frequency=checkpoint_frequency,
            ),
            callbacks=[
            ]
        ),
        param_space={
        }
    )

    return tuner

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

class BasePolicy(torch.nn.Module): #, ABC
    def __init__(self, input_size, hidden_size, lstm_layers=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

    #@abstractmethod
    def critic(self, hidden):
        #raise NotImplementedError
        return hidden

    def encode_observations(self, flat_observations):
        return hidden, lookup

    def decode_actions(self, flat_hidden, lookup):
        return logits

def _make_recurrent_policy(Policy):
    class Recurrent(Policy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert self.lstm_layers > 0
            self.lstm = pufferlib.torch.BatchFirstLSTM(
                self.input_size,
                self.hidden_size,
                self.lstm_layers,
            )

        def encode_observations(self, x, state, seq_lens):
            B, TT, _ = x.shape
            x = x.reshape(B*TT, -1)

            hidden, lookup = super().encode_observations(x)
            assert hidden.shape == (B*TT, self.hidden_size)

            hidden = hidden.view(B, TT, self.hidden_size)
            hidden, state = self.lstm(hidden, state)
            hidden = hidden.reshape(B*TT, self.hidden_size)

            return hidden, state, lookup
    return Recurrent


def make_rllib_policy(policy_cls, lstm_layers):
    assert issubclass(policy_cls, BasePolicy)

    if lstm_layers > 0:
        policy_cls = _make_recurrent_policy(policy_cls)

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
                hidden, state, lookup = self.encode_observations(x, state, seq_lens)
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

def make_cleanrl_policy(policy_cls, lstm_layers=0):
    assert issubclass(policy_cls, Base)
    if lstm_layers > 0:
        policy_cls = _make_recurrent_policy(policy_cls, lstm_layers)

    class CleanRLPolicy(Policy):
        def __init__(self, hidden_size, *args, **kwargs):
            super().__init__(*args, **kwargs)

        # TODO: Compute seq_lens from done
        def get_value(self, x, lstm_state=None, done=None):
            logits, _ = self.forward(x, lstm_state, done)
            return self.value_head(logits)

        # TODO: Compute seq_lens from done
        def get_action_and_value(self, x, lstm_state=None, done=None, action=None):
            logits, lstm_state = self.forward_rnn(x, lstm_state, done)
            value = self.value_head(x)

            mulit_categorical = [Categorical(logits=l) for l in flat_logits]

            if action is None:
                action = torch.stack([c.sample() for c in mulit_categorical])
            else:
                action = action.view(-1, action.shape[-1]).T

            logprob = torch.stack([c.log_prob(a) for c, a in zip(mulit_categorical, action)]).T.sum(1)
            entropy = torch.stack([c.entropy() for c in mulit_categorical]).T.sum(1)

            if lstm_layers > 0:
                return action.T, logprob, entropy, value, lstm_state
            return action.T, logprob.sum, entropy, value
   
    return CleanRLPolicy


class FCNetwork(TorchModelV2, torch.nn.Module):
    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def value_function(self):
        return self.value.view(-1)

    def encode_observations(self, flat_observations):
        return hidden, lookup

    def decode_actions(self, flat_hidden, lookup):
        return logits

    def forward(self, x, state, seq_lens):
        hidden, lookup = self.encode_observations(x['obs'])
        self.value = self.value_head(hidden)
        logits = self.decode_actions(hidden, lookup)

        return logits, state


class RecurrentNetwork(RLLibRecurrentNetwork, torch.nn.Module):
    def __init__(self, input_size, hidden_size, lstm_layers, *args, **kwargs):
        T()
        super().__init__(*args, **kwargs)
        torch.nn.Module.__init__(self)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.value_head = torch.nn.Linear(hidden_size, 1)
        self.lstm = pufferlib.torch.BatchFirstLSTM(input_size, hidden_size, lstm_layers)

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