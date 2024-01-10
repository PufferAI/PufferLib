from pdb import set_trace as T
import argparse
import sys
import os

import importlib
import inspect
import yaml

import pufferlib
import pufferlib.args
import pufferlib.utils
import pufferlib.models

from clean_pufferl import CleanPuffeRL, rollout, done_training


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

def make_policy(env, env_module, args):
    policy = env_module.Policy(env, **args.policy_args)
    if args.force_recurrence or env_module.Recurrent is not None:
        policy = pufferlib.models.RecurrentWrapper(
            env, policy, **args.recurrent_args)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(args.args.device)

def init_wandb(args, env_module):
    os.environ["WANDB_SILENT"] = "true"

    name = args.env
    if 'name' in args.env_args:
        name = args.env_args['name']

    import wandb
    wandb.init(
        id=args.exp_name or wandb.util.generate_id(),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config={
            'cleanrl': dict(args.args),
            'env': args.env_args,
            'policy': args.policy_args,
            'recurrent': args.recurrent_args,
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

def train(args, env_module, make_env):
    trainer = CleanPuffeRL(
        config=args.args,
        agent_creator=make_policy,
        agent_kwargs={'env_module': env_module, 'args': args},
        env_creator=make_env,
        env_creator_kwargs=args.env_args,
        vectorization=args.vectorization,
        exp_name=args.exp_name,
        track=args.track,
    )

    while not done_training(trainer):
        trainer.evaluate()
        trainer.train()

    print('Done training. Saving data...')
    trainer.close()
    print('Run complete')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse environment argument', add_help=False)
    parser.add_argument('--env', type=str, default='pokemon_red', help='Environment package name')
    parser.add_argument('--env-id', type=str, default=None, help='Name of specific environment within the environment pkg. Not applicable for packages with only one environment.')
    parser.add_argument('--train', action='store_true', help='Train')
    parser.add_argument('--sweep', action='store_true', help='WandB Train Sweep')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate')
    parser.add_argument('--eval-model-path', type=str, default=None, help='Path to model to evaluate')
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

    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    env = args['env']
    assert env in config, f'Environment {env} not found in config.yaml. Not all specific environments have their own config. For instance, to use a specific atari environment, specify --env atari --env-id BreakoutNoFrameskip-v4 or similar'
    env_config = config[env]

    env_id = args['env_id']
    if env_id is not None:
        if env_id in config:
            env_config = config[env_id]
        else:
            print(f'WARNING: {env_id} not found in config.yaml. Using {env} config.') 

    # TODO: Improve install checking with pkg_resources
    try:
        env_module = importlib.import_module(f'pufferlib.environments.{env}')
    except:
        pufferlib.utils.install_requirements(env)
        env_module = importlib.import_module(f'pufferlib.environments.{env}')

    train_args = config['train']
    train_args.update(env_config['train'])

    sweep_args = config['sweep']
    # TODO: Custom sweep args
    #sweep_args.update(env_config['sweep'])

    policy_args = get_init_args(env_module.Policy.__init__)
    policy_args.update(env_config['policy'])

    if env_id is not None:
        make_env = env_module.env_creator(env_id)
    else:
        make_env = env_module.env_creator()

    env_args = get_init_args(make_env)
    env_args.update(env_args)

    recurrent_args = {}
    recurrent = env_module.Recurrent
    if recurrent is not None:
        recurrent_args = dict(
            input_size=recurrent.input_size,
            hidden_size=recurrent.hidden_size,
            num_layers=recurrent.num_layers
        )

    config = pufferlib.namespace(
        args=train_args,
        env_args=env_args,
        policy_args=policy_args,
        recurrent_args=recurrent_args,
    )

    for name, sub_config in config.items():
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

    assert sum((args.train, args.sweep, args.evaluate)) == 1, 'Must specify exactly one of --train, --sweep, or --evaluate'
    if args.train:
        train(args, env_module, make_env)
        exit(0)
    elif args.sweep:
        sweep(args, env_module, make_env, sweep_args)
        exit(0)
    elif args.evaluate and args.env != 'pokemon_red':
        rollout(
            make_env,
            args.env_args,
            agent_creator=make_policy,
            agent_kwargs={'env_module': env_module, 'args': args},
            model_path=args.eval_model_path,
            device=args.args.device
        )
        exit(0)

    ### One-off demo for pokemon red
    import numpy as np
    import torch

    def make_pokemon_red_overlay(bg, counts):
        nonzero = np.where(counts > 0, 1, 0)
        scaled = np.clip(counts, 0, 1000) / 1000.0

        # Convert counts to hue map
        hsv = np.zeros((*counts.shape, 3))
        hsv[..., 0] = scaled*(240.0/360.0)
        hsv[..., 1] = nonzero
        hsv[..., 2] = nonzero

        # Convert the HSV image to RGB
        import matplotlib.colors as mcolors
        overlay = 255*mcolors.hsv_to_rgb(hsv)

        # Upscale to 16x16
        kernel = np.ones((16, 16, 1), dtype=np.uint8)
        overlay = np.kron(overlay, kernel).astype(np.uint8)
        mask = np.kron(nonzero, kernel[..., 0]).astype(np.uint8)
        mask = np.stack([mask, mask, mask], axis=-1).astype(bool)

        # Combine with background
        render = bg.copy().astype(np.int32)
        render[mask] = 0.2*render[mask] + 0.8*overlay[mask]
        render = np.clip(render, 0, 255).astype(np.uint8)
        return render

    env = make_env(**env_args)
    if args.exp_name is None:
        agent = make_policy(env, **policy_args)
    else:
        agent = torch.load(args.eval_model_path, map_location=args.args.device)

    terminal = truncated = True

    import cv2
    bg = cv2.imread('kanto_map_dsv.png')
 
    while True:
        if terminal or truncated:
            if args.args.verbose:
                print('---  Reset  ---')

            ob, info = env.reset()
            state = None
            step = 0
            return_val = 0

        ob = torch.tensor(ob).unsqueeze(0).to(args.args.device)
        with torch.no_grad():
            if hasattr(agent, 'lstm'):
                action, _, _, _, state = agent.get_action_and_value(ob, state)
            else:
                action, _, _, _ = agent.get_action_and_value(ob)

        ob, reward, terminal, truncated, _ = env.step(action[0].item())
        return_val += reward

        counts_map = env.env.counts_map
        if np.sum(counts_map) > 0 and step % 500 == 0:
            overlay = make_pokemon_red_overlay(bg, counts_map)
            cv2.imshow('Pokemon Red', overlay[1000:][::4, ::4])
            cv2.waitKey(1)

        if args.args.verbose:
            print(f'Step: {step} Reward: {reward:.4f} Return: {return_val:.2f}')

        if not env_args['headless']:
            env.render()

        step += 1
