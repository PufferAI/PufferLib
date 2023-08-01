from pdb import set_trace as T
import argparse
import sys
import subprocess

import pufferlib

# TODO: Fix circular import depending on import order
from clean_pufferl import CleanPuffeRL
import config as demo_config


def parse_arguments():
    '''Parse environment to train from command line'''
    parser = argparse.ArgumentParser(description="Parse environment argument")
    parser.add_argument("--env", type=str, default="nmmo", help="Environment name")
    args = parser.parse_args()
    return args.env

def install_requirements(env):
    '''Pip install dependencies for specified environment'''
    pip_install_cmd = [sys.executable, "-m", "pip", "install", "-e" f".[{env}]"]
    proc = subprocess.run(pip_install_cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(f"Error installing requirements: {proc.stderr}")

def train_model(binding):
    agent = pufferlib.frameworks.cleanrl.make_policy(
            config.Policy, recurrent_args=config.recurrent_args,
            recurrent_kwargs=config.recurrent_kwargs,
        )(binding, *config.policy_args, **config.policy_kwargs).to(config.device)

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

    try:
        install_requirements(env)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    config = demo_config.all[env]()
    binding = config.make_binding()
    train_model(binding)