#!/usr/bin/env python
"""Self-play training with PolicyPool for Dogfight.

This script wires up the PolicyPool for self-play opponent selection.
When the curriculum system achieves mastery of a stage, checkpoints are
automatically saved to the pool for future opponent selection.

Usage:
    # Basic self-play training
    python pufferlib/ocean/dogfight/train_selfplay.py

    # With wandb logging
    python pufferlib/ocean/dogfight/train_selfplay.py --wandb --wandb-project dogfight-selfplay

    # Resume from pool with existing checkpoints
    python pufferlib/ocean/dogfight/train_selfplay.py --pool-dir experiments/dogfight_pool

    # Start with empty pool (default)
    python pufferlib/ocean/dogfight/train_selfplay.py
"""
import os
import sys
import argparse

import pufferlib
import pufferlib.vector
from pufferlib import pufferl
from pufferlib.policy_pool import PolicyPool, setup_pool_callback
from pufferlib.ocean.dogfight.dogfight import Dogfight


# Default pool directory
DEFAULT_POOL_DIR = 'experiments/dogfight_pool'


def main():
    env_name = 'puffer_dogfight'

    # Extract --pool-dir from sys.argv before pufferl parses
    pool_dir = DEFAULT_POOL_DIR
    new_argv = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == '--pool-dir':
            if i + 1 < len(sys.argv):
                pool_dir = sys.argv[i + 1]
                i += 2
                continue
        elif sys.argv[i].startswith('--pool-dir='):
            pool_dir = sys.argv[i].split('=', 1)[1]
            i += 1
            continue
        new_argv.append(sys.argv[i])
        i += 1
    sys.argv = new_argv

    # Load standard dogfight config (now without --pool-dir)
    args = pufferl.load_config(env_name)

    # Extract obs_scheme from env config
    obs_scheme = args['env'].get('obs_scheme', 0)

    # Create pool directory if needed
    os.makedirs(pool_dir, exist_ok=True)

    # Create PolicyPool with matching obs_scheme
    pool = PolicyPool(pool_dir, obs_scheme=obs_scheme)
    print(f'[SELFPLAY] PolicyPool created: {pool_dir} (obs_scheme={obs_scheme}, {len(pool)} entries)')

    # Add pool to env kwargs
    args['env']['policy_pool'] = pool

    # Create environment using standard pufferl flow
    vecenv = pufferl.load_env(env_name, args)

    # Create policy
    policy = pufferl.load_policy(args, vecenv, env_name)

    # Create logger if requested
    logger = None
    if args['neptune']:
        logger = pufferl.NeptuneLogger(args)
    elif args['wandb']:
        logger = pufferl.WandbLogger(args)

    # Create trainer
    train_config = {**args['train'], 'env': env_name}
    trainer = pufferl.PuffeRL(train_config, vecenv, policy, logger)

    # Wire up pool callback (checkpoints saved on mastery)
    # vecenv.driver_env is the unwrapped Dogfight instance
    setup_pool_callback(vecenv.driver_env, trainer, pool)
    print(f'[SELFPLAY] Pool callback wired to trainer')

    # Standard training loop
    while trainer.global_step < train_config['total_timesteps']:
        if train_config['device'] == 'cuda':
            import torch
            torch.compiler.cudagraph_mark_step_begin()
        trainer.evaluate()
        if train_config['device'] == 'cuda':
            import torch
            torch.compiler.cudagraph_mark_step_begin()
        trainer.train()

        # Log pool status periodically
        if trainer.epoch % 100 == 0 and trainer.epoch > 0:
            pool_size = len(pool)
            if pool_size > 0:
                print(f'[SELFPLAY] Pool size: {pool_size} checkpoints')

    # Cleanup
    model_path = trainer.close()
    if logger:
        logger.close(model_path)

    print(f'[SELFPLAY] Training complete. Final pool size: {len(pool)}')
    print(f'[SELFPLAY] Pool directory: {pool_dir}')


if __name__ == '__main__':
    main()
