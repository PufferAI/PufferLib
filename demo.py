from pdb import set_trace as T
import argparse
import sys
import time
import os
import importlib
import inspect
from collections import defaultdict

import torch

import pufferlib

# TODO: Fix circular import depending on import order
from clean_pufferl import CleanPuffeRL

import pufferlib.utils
import pufferlib.models

import config


def get_init_args(fn):
    sig = inspect.signature(fn)
    
    # Extract the arguments and their default values
    args = {}
    for name, param in sig.parameters.items():
        if name in ('self', 'env'):
            continue
        args[name] = param.default if param.default is not inspect.Parameter.empty else None

    return args

def make_config(env):
    try:
        env_module = importlib.import_module(f'pufferlib.registry.{env}')
    except:
        pufferlib.utils.install_requirements(env)
        env_module = importlib.import_module(f'pufferlib.registry.{env}')

    cleanrl_init, cleanrl_train, sweep_config = getattr(config, env)()

    env_kwargs = get_init_args(env_module.make_env)
    policy_kwargs = get_init_args(env_module.Policy.__init__)

    recurrent_kwargs = {}
    if env_module.Recurrent is not None:
        recurrent_kwargs = get_init_args(env_module.Recurrent.__init__)

    return env_module, sweep_config, pufferlib.namespace(
        cleanrl_init = cleanrl_init,
        cleanrl_train = cleanrl_train,
        env_kwargs = env_kwargs,
        policy_kwargs = policy_kwargs,
        recurrent_kwargs = recurrent_kwargs,
   )
 
def make_policy(envs, env_module, args):
    policy = env_module.Policy(envs.driver_env, **args.policy_kwargs)
    if args.force_recurrence or env_module.Recurrent is not None:
        policy = pufferlib.models.RecurrentWrapper(
            envs, policy, **args.recurrent_kwargs)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(args.cleanrl_init['device'])


def init_wandb(args, env_module):
    os.environ["WANDB_SILENT"] = "true"
    import wandb
    wandb.init(
        id=args.run_id or wandb.util.generate_id(),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config={
            'env': args.env_kwargs,
            'policy': args.policy_kwargs,
            'recurrent': args.recurrent_kwargs,
            **args.cleanrl_init,
            **args.cleanrl_train,
        },
        name=args.env,
        monitor_gym=True,
        save_code=True,
        resume="allow",
    )
    return wandb.run.id

def sweep(args, env_module, sweep_config):
    import wandb
    sweep_id = wandb.sweep(sweep=sweep_config, project="pufferlib")

    def main():
        args.run_id = init_wandb(args, env_module)
        if hasattr(wandb.config, 'cleanrl_init'):
            args.cleanrl_init.update(wandb.config.cleanrl_init)
            wandb.config.update(wandb.config.cleanrl_init, allow_val_change=True)
        if hasattr(wandb.config, 'cleanrl_train'):
            args.cleanrl_train.update(wandb.config.cleanrl_train)
            wandb.config.update(wandb.config.train, allow_val_change=True)
        if hasattr(wandb.config, 'env'):
            args.env_kwargs.update(wandb.config.env)
        if hasattr(wandb.config, 'policy'):
            args.policy_kwargs.update(wandb.config.policy)
        #args.cleanrl_init['learning_rate'] = wandb.config.learning_rate
        train(args, env_module)

    wandb.agent(sweep_id, main, count=20)

def train(args, env_module):
    trainer = CleanPuffeRL(
        agent_creator=make_policy,
        agent_kwargs={'env_module': env_module, 'args': args},
        env_creator=env_module.make_env,
        env_creator_kwargs=args.env_kwargs,
        **args.cleanrl_init,
        run_id=args.run_id,
        track=args.track,
    )

    num_updates = (args.cleanrl_init['total_timesteps']
        // args.cleanrl_init['batch_size'])

    for update in range(num_updates):
        trainer.evaluate()
        trainer.train(**args.cleanrl_train)

    trainer.close()

def evaluate(args, env_module):
    env_creator = env_module.make_env
    env_creator_kwargs = args.env_kwargs
    env = env_creator(**env_creator_kwargs)

    import torch
    device = args.cleanrl_init['device']
    agent = torch.load('agent.pt').to(device)

    ob = env.reset()
    for i in range(100):
        ob = torch.tensor(ob).view(1, -1).to(device)
        with torch.no_grad():
            action  = agent.get_action_and_value(ob)[0][0].item()

        ob, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1)

        if done:
            ob = env.reset()
            print('---Reset---')
            env.render()
            time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse environment argument", add_help=False)
    parser.add_argument("--env", type=str, default="nmmo", help="Environment name")
    parser.add_argument("--mode", type=str, default="train", help="train/eval/sweep")
    parser.add_argument("--run-id", type=str, default=None, help="Experiment name")
    parser.add_argument('--wandb-entity', type=str, default='jsuarez', help='WandB entity')
    parser.add_argument('--wandb-project', type=str, default='pufferlib', help='WandB project')
    parser.add_argument('--wandb-group', type=str, default='debug', help='WandB group')
    parser.add_argument('--track', action='store_true', help='Track on WandB')
    parser.add_argument('--force-recurrence', action='store_true', help='Force model to be recurrent, regardless of defaults')

    clean_parser = argparse.ArgumentParser(parents=[parser])
    args = parser.parse_known_args()[0].__dict__
    env = args['env']

    env_module, sweep_config, cfg = make_config(env)
    for name, sub_config in cfg.items():
        args[name] = {}
        for key, value in sub_config.items():
            data_key = f'{name}.{key}'
            cli_key = f'--{data_key}'.replace('_', '-')
            parser.add_argument(cli_key, default=value)
            clean_parser.add_argument(cli_key, default=value, metavar='', help=env)
            args[name][key] = getattr(parser.parse_known_args()[0], data_key)

    clean_parser.parse_args(sys.argv[1:])
    args = pufferlib.namespace(**args)

    if args.mode == 'sweep':
        args.track = True
    elif args.track:
        args.run_id = init_wandb(args, env_module)

    if args.mode == 'train':
        train(args, env_module)
    elif args.mode == 'eval':
        evaluate(args, env_module)
    elif args.mode == 'sweep':
        sweep(args, env_module, sweep_config)
