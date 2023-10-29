from pdb import set_trace as T
import argparse
import sys
import os

import importlib
import inspect

import pufferlib
import pufferlib.args
import pufferlib.utils
import pufferlib.models

import config
from clean_pufferl import CleanPuffeRL


def get_init_args(fn):
    sig = inspect.signature(fn)
    args = {}
    for name, param in sig.parameters.items():
        if name in ('self', 'env'):
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        else:
            args[name] = param.default if param.default is not inspect.Parameter.empty else None
    return args

def make_config(env):
    # TODO: Improve install checking with pkg_resources
    try:
        env_module = importlib.import_module(f'pufferlib.registry.{env}')
    except:
        pufferlib.utils.install_requirements(env)
        env_module = importlib.import_module(f'pufferlib.registry.{env}')

    all_configs = config.all()
    args, sweep_config = all_configs[env]()

    env_kwargs = get_init_args(env_module.make_env)
    policy_kwargs = get_init_args(env_module.Policy.__init__)

    recurrent_kwargs = {}
    if env_module.Recurrent is not None:
        recurrent_kwargs = get_init_args(env_module.Recurrent.__init__)

    return env_module, sweep_config, pufferlib.namespace(
        args=args,
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

    if args.args.cuda:
        return policy.cuda()

    return policy

def init_wandb(args, env_module):
    os.environ["WANDB_SILENT"] = "true"

    name = args.env
    if 'name' in args.env_kwargs:
        name = args.env_kwargs['name']

    import wandb
    wandb.init(
        id=args.exp_name or wandb.util.generate_id(),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config={
            'cleanrl': dict(args.args),
            'env': args.env_kwargs,
            'policy': args.policy_kwargs,
            'recurrent': args.recurrent_kwargs,
        },
        name=name,
        monitor_gym=True,
        save_code=True,
        resume="allow",
    )
    return wandb.run.id

def sweep(args, env_module, sweep_config):
    import wandb
    sweep_id = wandb.sweep(sweep=sweep_config, project="pufferlib")

    def main():
        args.exp_name = init_wandb(args, env_module)
        if hasattr(wandb.config, 'cleanrl'):
            # TODO: Add update method to namespace
            args.args.__dict__.update(wandb.config.cleanrl)
        if hasattr(wandb.config, 'env'):
            args.env_kwargs.update(wandb.config.env)
        if hasattr(wandb.config, 'policy'):
            args.policy_kwargs.update(wandb.config.policy)
        train(args, env_module)

    wandb.agent(sweep_id, main, count=20)

def train(args, env_module):
    trainer = CleanPuffeRL(
        config=args.args,
        agent_creator=make_policy,
        agent_kwargs={'env_module': env_module, 'args': args},
        env_creator=env_module.make_env,
        env_creator_kwargs=args.env_kwargs,
        vectorization=args.vectorization,
        exp_name=args.exp_name,
        track=args.track,
    )

    for update in range(trainer.total_updates):
        trainer.evaluate()
        trainer.train()

    trainer.close()

def evaluate(args, env_module):
    env_creator = env_module.make_env
    env_creator_kwargs = args.env_kwargs
    env_creator_kwargs['headless'] = False
    env_creator_kwargs['save_video'] = True
    env = env_creator(**env_creator_kwargs)

    import torch
    device = 'cuda'
    agent = torch.load('pokemon_red_50m/model_001525.pt').to(device)

    while True:
        ob, info = env.reset()
        return_val = 0
        for i in range(20480):
            ob = torch.tensor(ob).unsqueeze(0).to(device)
            with torch.no_grad():
                action  = agent.get_action_and_value(ob)[0][0].item()

            ob, reward, terminal, truncated, _ = env.step(action)
            return_val += reward
            env.render()

            if terminal or truncated:
                ob, info = env.reset()
                print('---Reset---')
                env.render()

        print('Return:', return_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse environment argument", add_help=False)
    parser.add_argument('--env', type=str, default="nmmo", help="Environment name")
    parser.add_argument('--mode', type=str, default="train", help="train/eval/sweep")
    parser.add_argument('--exp-name', type=str, default=None, help="Experiment name")
    parser.add_argument('--vectorization', type=str, default='serial', help='Vectorization method (serial, multiprocessing, ray)')
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
            parser.add_argument(cli_key, default=value, type=type(value))
            clean_parser.add_argument(cli_key, default=value, metavar='', help=env)
            args[name][key] = getattr(parser.parse_known_args()[0], data_key)

    clean_parser.parse_args(sys.argv[1:])
    args['args'] = pufferlib.args.CleanPuffeRL(**args['args'])
    args = pufferlib.namespace(**args)
    vec = args.vectorization
    assert vec in 'serial multiprocessing ray'.split()
    if vec == 'serial':
        args.vectorization = pufferlib.vectorization.Serial
    elif vec == 'multiprocessing':
        args.vectorization = pufferlib.vectorization.Multiprocessing
    elif vec == 'ray':
        args.vectorization = pufferlib.vectorization.Ray

    if args.mode == 'sweep':
        args.track = True
    elif args.track:
        args.exp_name = init_wandb(args, env_module)

    if args.mode == 'train':
        train(args, env_module)
    elif args.mode == 'eval':
        evaluate(args, env_module)
    elif args.mode == 'sweep':
        sweep(args, env_module, sweep_config)
