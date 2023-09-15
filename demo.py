from pdb import set_trace as T
import argparse
import sys
import subprocess

import pufferlib

# TODO: Fix circular import depending on import order
from clean_pufferl import CleanPuffeRL

import pufferlib.utils
import pufferlib.models


def parse_arguments():
    '''Parse environment to train from command line'''
    parser = argparse.ArgumentParser(description="Parse environment argument")
    parser.add_argument("--env", type=str, default="nmmo", help="Environment name")
    args = parser.parse_args()
    return args.env

def make_policy(envs, config):
    policy = config.policy_cls(envs.driver_env, **config.policy_kwargs)
    if config.recurrent_cls is not None:
        policy = pufferlib.models.RecurrentWrapper(
            envs, policy, **config.recurrent_kwargs)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)
    policy = policy.to(config.cleanrl_init.device)
    return policy

def train_model(config, env_creator):
    env_creator_kwargs = config.env_creators[env_creator]
    trainer = CleanPuffeRL(
        agent_creator=make_policy,
        agent_kwargs={'config': config},
        env_creator=env_creator,
        env_creator_kwargs=env_creator_kwargs,
        **config.cleanrl_init.dict(),
    )

    #trainer.load_model(path)

    num_updates = (config.cleanrl_init.total_timesteps 
        // config.cleanrl_init.batch_size)

    for update in range(num_updates):
        print("Evaluating...", update)
        trainer.evaluate()
        trainer.train(**config.cleanrl_train.dict())

    trainer.close()


if __name__ == '__main__':
    env = parse_arguments()

    import config as config_module
    config = getattr(config_module, env)()

    for env_creator in config.env_creators:
        train_model(config, env_creator)
