from pdb import set_trace as T
import numpy as np

from ray.train.rl.rl_predictor import RLPredictor as RLlibPredictor
from ray.tune.registry import register_env as tune_register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env import ParallelPettingZooEnv

def register_env(name, env_creator):
    assert type(name) == str, 'Name must be a str'
    tune_register_env(name, lambda config: ParallelPettingZooEnv(env_creator())) 

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
        self._on_episode_end(worker, policies, episode, **kwargs)
        return super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            **kwargs
        )