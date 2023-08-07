from pdb import set_trace as T
import argparse
import sys
import subprocess

import pufferlib

# TODO: Fix circular import depending on import order
from clean_pufferl import CleanPuffeRL
import config

import pufferlib.utils


def parse_arguments():
    '''Parse environment to train from command line'''
    parser = argparse.ArgumentParser(description="Parse environment argument")
    parser.add_argument("--env", type=str, default="nmmo", help="Environment name")
    args = parser.parse_args()
    return args.env

def train_model(args):
    agent = pufferlib.frameworks.cleanrl.make_policy(
            config.policy_cls,
            recurrent_kwargs=config.recurrent_kwargs,
        )(**config.policy_kwargs).to(config.policy_kwargs.device)

    trainer = CleanPuffeRL(
            binding,
            agent,
            num_buffers=config.num_buffers,
            num_envs=config.num_envs,
            num_cores=config.num_cores,
            batch_size=config.batch_size,
            vec_backend=config.vec_backend,
            seed=config.seed,
    )

    #trainer.load_model(path)
    #trainer.init_wandb()

    num_updates = config.total_timesteps // config.batch_size
    for update in range(num_updates):
        print("Evaluating...", update)
        trainer.evaluate()
        trainer.train(batch_rows=config.batch_rows, bptt_horizon=config.bptt_horizon)

    trainer.close()


if __name__ == '__main__':
    env = parse_arguments()

    args = getattr(config, env)
    train_model(args)