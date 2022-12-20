from pdb import set_trace as T

import pufferlib

import ray
import pufferlib
from ray.air import CheckpointConfig
from ray.air.config import RunConfig, ScalingConfig
from ray.tune.tuner import Tuner
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.train.rl.rl_trainer import RLTrainer
from ray.rllib.models import ModelCatalog


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
    '''Creates an RLlib tuner with sane defaults'''

    ray.init(
        include_dashboard=False, # WSL Compatibility
        ignore_reinit_error=True,
        num_gpus=num_gpus,
    )

    env_cls = binding.env_cls
    env_args = binding.env_args
    name = binding.env_name

    policy = pufferlib.rllib.make_rllib_policy(binding.policy,
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

if __name__ == '__main__':
    from environments import bindings
    for binding in bindings.values():
        tuner = make_rllib_tuner(binding)
        result = tuner.fit()[0]
        print('Saved ', result.checkpoint)