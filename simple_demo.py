'''The main demo script includes yaml configs for all environments,
dynamic loading of environments, and other advanced features. If you
just want to run on a single environment, this is a simpler option.'''

from pdb import set_trace as T
import argparse
import os

import torch

import pufferlib
import pufferlib.vector
import pufferlib.frameworks.cleanrl

from rich_argparse import RichHelpFormatter
from rich.console import Console
from rich.traceback import install
install(show_locals=False)

import clean_pufferl

def make_policy(env, use_rnn):
    '''Make the policy for the environment'''
    policy = Policy(env)
    if use_rnn:
        policy = Recurrent(env, policy)
        return pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        return pufferlib.frameworks.cleanrl.Policy(policy)

def train(args):
    args.wandb = None
    if args.track:
        args.wandb = init_wandb(args, args.env, id=args.train.exp_id)
        args.train.__dict__.update(dict(args.wandb.config.train))
    if args.vec.backend == 'serial':
        backend = pufferlib.vector.Serial
    elif args.vec.backend == 'multiprocessing':
        backend = pufferlib.vector.Multiprocessing
    elif args.vec == 'ray':
        backend = pufferlib.vector.Ray
    else:
        raise ValueError(f'Invalid --vec.backend (serial/multiprocessing/ray).')

    vecenv = pufferlib.vector.make(
        make_env,
        num_envs=args.vec.num_envs,
        num_workers=args.vec.num_workers,
        batch_size=args.vec.env_batch_size,
        zero_copy=args.vec.zero_copy,
        backend=backend,
    )
    policy = make_policy(vecenv.driver_env, args.use_rnn).to(args.train.device)

    args.train.env = args.env
    data = clean_pufferl.create(args.train, vecenv, policy, wandb=args.wandb)
    while data.global_step < args.train.total_timesteps:
        try:
            clean_pufferl.evaluate(data)
            clean_pufferl.train(data)
        except KeyboardInterrupt:
            clean_pufferl.close(data)
            os._exit(0)
        except Exception:
            Console().print_exception()
            os._exit(0)

    clean_pufferl.evaluate(data)
    clean_pufferl.close(data)

def init_wandb(args, name, id=None, resume=True):
    import wandb
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config={
            'train': dict(args.train),
            'vec': dict(args.vec),
        },
        name=name,
        monitor_gym=True,
        save_code=True,
        resume=resume,
    )
    return wandb

def sweep(args):
    import wandb
    sweep_id = wandb.sweep(
        sweep=dict(
            method='random',
            name=sweep,
            metric=dict(
                goal='maximize',
                name='environment/episode_return',
            ),
            parameters=dict(
                learning_rate=dict(
                    distribution='log_uniform_values',
                    min=1e-4,
                    max=1e-1
                ),
                batch_size=dict(
                    values=[512, 1024, 2048],
                ),
                minibatch_size=dict(
                    values=[128, 256, 512],
                ),
                bptt_horizon=dict(
                    values=[4, 8, 16],
                ),
            ),
        ),
        project="pufferlib",
    )

    args.track = True
    wandb.agent(sweep_id, lambda: train(args), count=100)

if __name__ == '__main__':
    # TODO: Add check against old args like --config to demo
    parser = argparse.ArgumentParser(
            description=f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]'
        ' demo options. Shows valid args for your env and policy',
        formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--mode', type=str, default='train', choices='train eval evaluate sweep autotune baseline profile'.split())
    parser.add_argument('--use-rnn', action='store_true')
    parser.add_argument('--eval-model-path', type=str, default=None, help='Path to model to evaluate')
    parser.add_argument('--baseline', action='store_true', help='Baseline run')
    parser.add_argument('--no-render', action='store_true', help='Disable render during evaluate')
    parser.add_argument('--wandb-entity', type=str, default='jsuarez', help='WandB entity')
    parser.add_argument('--wandb-project', type=str, default='pufferlib', help='WandB project')
    parser.add_argument('--wandb-group', type=str, default='debug', help='WandB group')
    parser.add_argument('--track', action='store_true', help='Track on WandB')

    # Train configuration
    parser.add_argument('--train.exp-id', type=str, default=None)
    parser.add_argument('--train.seed', type=int, default=1)
    parser.add_argument('--train.torch-deterministic', action='store_true')
    parser.add_argument('--train.cpu-offload', action='store_true')
    parser.add_argument('--train.device', type=str, default='cuda'
        if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train.total-timesteps', type=int, default=10_000_000)
    parser.add_argument('--train.learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--train.anneal-lr', action='store_false')
    parser.add_argument('--train.gamma', type=float, default=0.99)
    parser.add_argument('--train.gae-lambda', type=float, default=0.95)
    parser.add_argument('--train.update-epochs', type=int, default=4)
    parser.add_argument('--train.norm-adv', action='store_true')
    parser.add_argument('--train.clip-coef', type=float, default=0.1)
    parser.add_argument('--train.clip-vloss', action='store_false')
    parser.add_argument('--train.ent-coef', type=float, default=0.01)
    parser.add_argument('--train.vf-coef', type=float, default=0.5)
    parser.add_argument('--train.vf-clip-coef', type=float, default=0.1)
    parser.add_argument('--train.max-grad-norm', type=float, default=0.5)
    parser.add_argument('--train.target-kl', type=float, default=None)
    parser.add_argument('--train.checkpoint-interval', type=int, default=200)
    parser.add_argument('--train.batch-size', type=int, default=1024)
    parser.add_argument('--train.minibatch-size', type=int, default=512)
    parser.add_argument('--train.bptt-horizon', type=int, default=16)
    parser.add_argument('--train.compile', action='store_true')
    parser.add_argument('--train.compile-mode', type=str, default='reduce-overhead')

    parser.add_argument('--vec.backend', type=str, default='multiprocessing',
        choices='serial multiprocessing ray'.split())
    parser.add_argument('--vec.num-envs', type=int, default=8)
    parser.add_argument('--vec.num-workers', type=int, default=8)
    parser.add_argument('--vec.env-batch-size', type=int, default=8)
    parser.add_argument('--vec.zero-copy', action='store_true')
    parsed = parser.parse_args()

    args = {}
    for k, v in vars(parsed).items():
        if '.' in k:
            group, name = k.split('.')
            if group not in args:
                args[group] = {}

            args[group][name] = v
        else:
            args[k] = v

    args['train'] = pufferlib.namespace(**args['train'])
    args['vec'] = pufferlib.namespace(**args['vec'])
    args = pufferlib.namespace(**args)

    # Import your environment and policy
    from pufferlib.environments.atari import Policy, Recurrent, env_creator
    make_env = env_creator(args.env)

    if args.mode == 'train':
        train(args)
    elif args.mode in ('eval', 'evaluate'):
        try:
            clean_pufferl.rollout(
                make_env,
                env_kwargs={},
                agent_creator=make_policy,
                agent_kwargs={'use_rnn': args.use_rnn},
                model_path=args.eval_model_path,
                device=args.train.device
            )
        except KeyboardInterrupt:
            os._exit(0)
    elif args.mode == 'sweep':
        sweep(args)
    elif args.mode == 'autotune':
        pufferlib.vector.autotune(make_env, batch_size=args.vec.env_batch_size)
