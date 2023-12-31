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

from clean_pufferl import CleanPuffeRL, rollout


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
    import yaml
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    train_defaults = config['train']
    sweep_defaults = config['sweep']

    assert env in config
    env_config = config[env]

    # Some envs have a different package name than the env name
    package = env_config['package']
    if package is None:
        package = env

    # TODO: Improve install checking with pkg_resources
    try:
        env_module = importlib.import_module(f'pufferlib.environments.{package}')
    except:
        pufferlib.utils.install_requirements(package)
        env_module = importlib.import_module(f'pufferlib.environments.{package}')

    train_defaults.update(env_config['train'])
    train_args = train_defaults

    policy_args = env_config['policy']
    env_args = env_config['env']

    env_kwargs = get_init_args(env_module.make_env)
    env_kwargs.update(env_args)

    policy_kwargs = get_init_args(env_module.Policy.__init__)
    policy_kwargs.update(policy_args)

    recurrent_kwargs = {}
    recurrent = env_module.Recurrent
    if recurrent is not None:
        recurrent_kwargs = dict(
            input_size=recurrent.input_size,
            hidden_size=recurrent.hidden_size,
            num_layers=recurrent.num_layers
        )

    return env_module, sweep_defaults, pufferlib.namespace(
        args=train_args,
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

    return policy.to(args.args.device)

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
        resume=True,
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
    env = env_creator(**env_creator_kwargs)

    import torch
    device = args.args.device
    agent = torch.load(args.evaluate, map_location=device)
    terminal = truncated = True

    while True:
        if terminal or truncated:
            print('---  Reset  ---')
            ob, info = env.reset()
            state = None
            step = 0
            return_val = 0

        ob = torch.tensor(ob).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(agent, 'lstm'):
                action, _, _, _, state = agent.get_action_and_value(ob, state)
            else:
                action, _, _, _ = agent.get_action_and_value(ob)
        
        ob, reward, terminal, truncated, _ = env.step(action[0].item())
        return_val += reward

        print(f'Step: {step} Reward: {reward:.4f} Return: {return_val:.2f}')
        env.render()
        step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse environment argument', add_help=False)
    parser.add_argument('--env', type=str, default='pokemon_red', help='Environment name')
    parser.add_argument('--train', action='store_true', help='Train')
    parser.add_argument('--sweep', action='store_true', help='WandB Train Sweep')
    parser.add_argument('--evaluate', type=str, help='Path to your .pt file')
    parser.add_argument('--no-render', action='store_true', help='Disable render during evaluate')
    parser.add_argument('--exp-name', type=str, default=None, help="Resume from experiment")
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
            if isinstance(value, bool) and value is False:
                action = 'store_false'
                parser.add_argument(cli_key, default=value, action='store_true')
                clean_parser.add_argument(cli_key, default=value, action='store_true')
            elif isinstance(value, bool) and value is True:
                data_key = f'{name}.no_{key}'
                cli_key = f'--{data_key}'.replace('_', '-')
                parser.add_argument(cli_key, default=value, action='store_false')
                clean_parser.add_argument(cli_key, default=value, action='store_false')
            else:
                parser.add_argument(cli_key, default=value, type=type(value))
                clean_parser.add_argument(cli_key, default=value, metavar='', type=type(value))

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

    if args.sweep:
        args.track = True
    elif args.track:
        args.exp_name = init_wandb(args, env_module)

    assert sum((args.train, args.sweep, args.evaluate is not None)) == 1, 'Must specify exactly one of --train, --sweep, or --evaluate'
    if args.train:
        train(args, env_module)
    elif args.sweep:
        sweep(args, env_module, sweep_config)
    else:
        rollout(env_module.make_env, args.env_kwargs,
            args.evaluate, device=args.args.device)
