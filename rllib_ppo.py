# Basic integration of PufferLib envs with RLlib
# This runs but is not recommended. The RLlib 2.0
# API is extremely buggy and has almost no error checking.
# Later versions pin a very recent version of Gym that breaks
# all compatibility with most popular RL environments.
# As a result, CleanRL is the priority for our early development
# Let us know if you want this expanded!

from pdb import set_trace as T

import ray
from ray.air import CheckpointConfig
from ray.air.config import RunConfig, ScalingConfig
from ray.tune.tuner import Tuner
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.registry import register_env
from ray.train.rl.rl_trainer import RLTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.env.multi_agent_env import make_multi_agent

import pufferlib.registry
import pufferlib.utils
import pufferlib.frameworks.rllib
import pufferlib.vectorization

FRAMESTACK = 4


def make_rllib_tuner(
        env_creator,
        policy_cls,
        name,
        *,
        algorithm='PPO',
        num_gpus=1,
        num_workers=1,
        num_envs_per_worker=1,
        rollout_fragment_length=16,
        train_batch_size=2**10,
        sgd_minibatch_size=32,
        num_sgd_iter=1,
        max_seq_len=16,
        training_steps=3,
        checkpoints_to_keep=5,
        checkpoint_frequency=1,):
    '''Creates an RLlib tuner with sane defaults'''

    # We need a dummy env with some properties for the policy
    test_env = pufferlib.vectorization.Serial(
        env_creator=env_creator,
        env_args=[name, FRAMESTACK],
        num_workers=1,
        envs_per_worker=1,
    )

    rllib_env_creator = lambda _ : env_creator(name, framestack=FRAMESTACK)

    class RLlibPolicy(TorchModelV2, policy_cls):
        def __init__(self, *args, **kwargs):
            policy_cls.__init__(self, **kwargs)
            TorchModelV2.__init__(self, *args)

        def value_function(self):
            return self.value.view(-1)

        def forward(self, x, state, seq_lens):
            hidden, lookup = self.encode_observations(x['obs'].float())
            logits, self.value = self.decode_actions(hidden, lookup)
            return logits, state

    ray.init(
        include_dashboard=False, # WSL Compatibility
        ignore_reinit_error=True,
        num_gpus=num_gpus,
    )

    #policy = pufferlib.frameworks.rllib.Policy(policy)
    ModelCatalog.register_custom_model(name, RLlibPolicy)
    register_env(name, rllib_env_creator)

    trainer = RLTrainer(
        algorithm=algorithm,
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=num_gpus>0
        ),
        config={
            'num_gpus': num_gpus,
            'num_workers': num_workers,
            'num_envs_per_worker': num_envs_per_worker,
            'rollout_fragment_length': rollout_fragment_length,
            'train_batch_size': train_batch_size,
            'sgd_minibatch_size': sgd_minibatch_size,
            'num_sgd_iter': num_sgd_iter,
            'framework': 'torch',
            'env': name,
            'model': {
                'custom_model': name,
                'custom_model_config': {
                    'envs': test_env,
                    'framestack': FRAMESTACK,
                    'flat_size': 64*7*7,
                },
                'max_seq_len': max_seq_len,
            },
        }
    )

    tuner = Tuner(
        trainer,
        _tuner_kwargs={'checkpoint_at_end': True},
        run_config=RunConfig(
            local_dir='results',
            verbose=1,
            stop={
                'training_iteration': training_steps
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
    import pufferlib.registry.atari
    env_name = 'BreakoutNoFrameskip-v4'

    env_creator = pufferlib.registry.atari.make_env
    policy_cls = pufferlib.models.Convolutional

    tuner = make_rllib_tuner(env_creator, policy_cls, env_name)
    result = tuner.fit()[0]
    print('Saved ', result.checkpoint)
