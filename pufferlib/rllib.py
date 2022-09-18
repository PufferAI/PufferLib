from pdb import set_trace as T
import numpy as np

import os

from ray.train.rl.rl_predictor import RLPredictor as RLlibPredictor
from ray.tune.registry import register_env as tune_register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import ParallelPettingZooEnv


def register_env(env_creator, name):
    tune_register_env(name, lambda config: ParallelPettingZooEnv(env_creator())) 

class RLPredictor(RLlibPredictor):
    def predict(self, data, **kwargs):
        batch = data.shape[0]
        #data = data.reshape(batch, -1)
        data = data.squeeze()
        result = super().predict(data, **kwargs)
        return result
        result = np.concatenate(list(result.values())).reshape(1, -1)
        return result

class Callbacks(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, trainer, **kwargs) -> None:
        '''Run after 1 epoch at the trainer level'''
        '''
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')

        if not hasattr(self, 'tournament'):
            print('Making Tournament')
            self.tournament = utils.RemoteTournament()
            self.tournament.async_from_path(
                'checkpoints', num_policies,
                env_creator, policy_mapping_fn)
        '''

        '''
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')

        idx = 1
        paths = os.listdir('checkpoints') 
        if paths:
            idx = 1 + max(int(e.split('-')[1]) for e in paths)

        os.mkdir(f'checkpoints/checkpoint-{idx}')
        trainer.save_checkpoint(f'checkpoints/checkpoint-{idx}')
        '''

        return super().on_train_result(
            algorithm=algorithm,
            result=result,
            trainer=trainer,
            **kwargs
        )

    '''
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        self._on_episode_end(worker, policies, episode, **kwargs)
        return super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            **kwargs
        )
    '''