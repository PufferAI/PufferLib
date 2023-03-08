# TODO: Update and fix this test

from pdb import set_trace as T

import os

import ray
from ray.air import CheckpointConfig
from ray.air.config import RunConfig
from ray.air.config import ScalingConfig  
from ray.tune.tuner import Tuner
from ray.train.rl import RLCheckpoint
from ray.train.rl.rl_trainer import RLTrainer

from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils.conversions import aec_to_parallel

import pufferlib
import pufferlib.evaluation
import pufferlib.frameworks
import pufferlib.frameworks.rllib
import pufferlib.utils


def test_tournament():
    ray.init(include_dashboard=False, num_gpus=1)

    NUM_POLICIES = 3

    env_creator = lambda: aec_to_parallel(knights_archers_zombies_v10.env(max_zombies=0))
    pufferlib.frameworks.rllib.register_env('kaz', env_creator)

    trainer = RLTrainer(
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
        algorithm="PPO",
        config={
            "num_gpus": 0,
            "num_workers": 4,
            "num_envs_per_worker": 1,
            "train_batch_size": 2**10,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 1,
            "framework": "torch",
            "env": "kaz",
            "multiagent": {
                "count_steps_by": "agent_steps"
            },
            "model": {
            },
        }
    )

    results_dir = 'results'
    tune_dir= 'Tune-'+pufferlib.utils.current_datetime()
    tuner = Tuner(
        trainer,
        _tuner_kwargs={"checkpoint_at_end": True},
        run_config=RunConfig(
            local_dir=results_dir,
            name=tune_dir,
            verbose=0,
            stop={"training_iteration": 10},
            checkpoint_config=CheckpointConfig(
                num_to_keep=None,
                checkpoint_frequency=1,
            ),
        ),
    )

    result = tuner.fit()[0]
    checkpoint = RLCheckpoint.from_checkpoint(result.checkpoint)

    def policy_mapping_fn(key: str, episode: int):
        key = key.split('_')[0]
        return hash(key) % NUM_POLICIES

    tournament = pufferlib.evaluation.LocalTournament(NUM_POLICIES, env_creator, policy_mapping_fn)
    tournament.server(os.path.join(results_dir, tune_dir))

    for i in range(NUM_POLICIES + 1):
        tournament.add(f'policy_{i}', checkpoint, anchor=i==0)

    tournament.remove(f'policy_{i}')

    for episode in range(20):
        ratings = tournament.run_match(episode)
        print(ratings)


if __name__ == '__main__':
    test_tournament()