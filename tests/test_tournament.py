from pdb import set_trace as T

import ray
from ray.air import CheckpointConfig
from ray.air.config import RunConfig
from ray.air.config import ScalingConfig  
from ray.tune.registry import register_env
from ray.tune.tuner import Tuner
from ray.train.rl import RLCheckpoint
from ray.train.rl.rl_trainer import RLTrainer

from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils.conversions import aec_to_parallel

import pufferlib


def test_tournament():
    ray.init(include_dashboard=False, num_gpus=1)

    NUM_POLICIES = 3

    env_creator = lambda: aec_to_parallel(knights_archers_zombies_v10.env(max_zombies=0))
    pufferlib.rllib.register_env('kaz', env_creator)

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

    tuner = Tuner(
        trainer,
        _tuner_kwargs={"checkpoint_at_end": True},
        run_config=RunConfig(
            local_dir='results',
            verbose=0,
            stop={"training_iteration": 15},
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_frequency=1,
            ),
        ),
        param_space={
            'callbacks': pufferlib.rllib.Callbacks,
        },
    )

    result = tuner.fit()[0]
    checkpoint = RLCheckpoint.from_checkpoint(result.checkpoint)

    def policy_mapping_fn(key: str, episode: int):
        key = key.split('_')[0]
        return hash(key) % NUM_POLICIES

    tournament = pufferlib.evaluation.LocalTournament(NUM_POLICIES, env_creator, policy_mapping_fn)

    for i in range(NUM_POLICIES + 1):
        tournament.add(f'policy_{i}', checkpoint, anchor=i==0)

    tournament.remove(f'policy_{i}')

    for episode in range(3):
        ratings = tournament.run_match(episode)
        print(ratings)

    #tournament.server('results/AIRPPO_2022-09-17_22-11-25/AIRPPO_32dc0_00000_0_2022-09-17_22-11-25/')

if __name__ == '__main__':
    test_tournament()