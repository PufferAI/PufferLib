import configparser
import argparse
import shutil
import glob
import ast
import os

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.frameworks.cleanrl

from rich_argparse import RichHelpFormatter
from rich.console import Console
from rich.traceback import install
install(show_locals=False) # Rich tracebacks

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

import clean_pufferl
   
def make_policy(env, policy_cls, rnn_cls, args):
    policy = policy_cls(env, **args['policy'])
    if rnn_cls is not None:
        policy = rnn_cls(env, policy, **args['rnn'])
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(args['train']['device'])

def init_wandb(args, id=None, resume=True):
    import wandb
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args['wandb_project'],
        entity=args['wandb_entity'],
        group=args['wandb_group'],
        name=args['env'],
        save_code=True,
        resume=resume,
        config=args,
    )
    return wandb

def sweep(args, make_env, policy_cls, rnn_cls):
    import wandb
    sweep_id = wandb.sweep(sweep=args['sweep'], project=args['wandb_project'])

    def main():
        try:
            wandb = init_wandb(args, id=args.exp_id)
            args['train'].update(wandb.config.train)
            train(args, make_env, policy_cls, rnn_cls, wandb)
        except Exception as e:
            Console().print_exception()

    wandb.agent(sweep_id, main, count=100)

def train(args, make_env, policy_cls, rnn_cls, wandb):
    if args['vec'] == 'serial':
        vec = pufferlib.vector.Serial
    elif args['vec'] == 'multiprocessing':
        vec = pufferlib.vector.Multiprocessing
    elif args['vec'] == 'ray':
        vec = pufferlib.vector.Ray
    else:
        raise ValueError(f'Invalid --vector (serial/multiprocessing/ray).')

    vecenv = pufferlib.vector.make(
        make_env,
        env_kwargs=None,#args.env,
        num_envs=args['train']['num_envs'],
        num_workers=args['train']['num_workers'],
        batch_size=args['train']['env_batch_size'],
        zero_copy=args['train']['zero_copy'],
        backend=vec,
    )
    policy = make_policy(vecenv.driver_env, policy_cls, rnn_cls, args)
    train_config = pufferlib.namespace(**args['train'],
        exp_id=args['exp_id'], env=args['env'])
    data = clean_pufferl.create(train_config, vecenv, policy, wandb=wandb)
    while data.global_step < train_config.total_timesteps:
        clean_pufferl.evaluate(data)
        clean_pufferl.train(data)

    stats, _ = clean_pufferl.evaluate(data)
    clean_pufferl.close(data)
    return stats, data.profile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]'
        ' demo options. Shows valid args for your env and policy',
        formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--default-config', default='config/default.ini')
    parser.add_argument('--config', default='config/ocean/grid.ini')
    parser.add_argument('--env', '--environment', type=str,
        default='squared', help='Name of specific environment to run')
    parser.add_argument('--mode', type=str, default='train',
        choices='train eval evaluate sweep sweep-carbs autotune profile'.split())
    parser.add_argument('--eval-model-path', type=str, default=None)
    parser.add_argument('--baseline', action='store_true',
        help='Pretrained baseline where available')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'None'])
    parser.add_argument('--vec', '--vector', '--vectorization', type=str,
        default='serial', choices=['serial', 'multiprocessing', 'ray'])
    parser.add_argument('--exp-id', '--exp-name', type=str,
        default=None, help="Resume from experiment")
    parser.add_argument('--wandb-entity', type=str, default='jsuarez')
    parser.add_argument('--wandb-project', type=str, default='pufferlib')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--track', action='store_true', help='Track on WandB')
    args = parser.parse_known_args()[0]

    if not os.path.exists(args.default_config):
        raise Exception(f'Default config {args.default_config} not found')

    for path in glob.glob('config/**/*.ini', recursive=True):
        p = configparser.ConfigParser()
        p.read(args.default_config)
        p.read(path)
        if args.env in p['base']['env_name'].split():
            break
    else:
        raise Exception('No config for env_name {}'.format(args.env))

    for section in p.sections():
        for key in p[section]:
            argparse_key = f'--{section}.{key}'.replace('_', '-')
            parser.add_argument(argparse_key, default=p[section][key])

    args = {'policy': {}, 'rnn': {}}
    for key, value in parser.parse_args().__dict__.items():
        next = args
        for subkey in key.split('.'):
            if subkey not in next:
                next[subkey] = {}
            prev = next
            next = next[subkey]
        try:
            prev[subkey] = ast.literal_eval(value)
        except:
            prev[subkey] = value

    import importlib
    env_module = importlib.import_module(
        f'pufferlib.environments.{args["base"]["package"]}')
    make_env = env_module.env_creator(args['env'])
    policy_cls = getattr(env_module.torch, args['base']['policy_name'])
    rnn_cls = getattr(env_module, args['base']['rnn_name'])

    if args['baseline']:
        assert args['mode'] in ('train', 'eval', 'evaluate')
        args['track'] = True
        version = '.'.join(pufferlib.__version__.split('.')[:2])
        args.exp_id = f'puf-{version}-{args.env_name}'
        args.wandb_group = f'puf-{version}-baseline'
        shutil.rmtree(f'experiments/{args.exp_id}', ignore_errors=True)
        run = init_wandb(args, args['exp_id'], resume=False)
        if args.mode in ('eval', 'evaluate'):
            model_name = f'puf-{version}-{args.env_name}_model:latest'
            artifact = run.use_artifact(model_name)
            data_dir = artifact.download()
            model_file = max(os.listdir(data_dir))
            args.eval_model_path = os.path.join(data_dir, model_file)
    if args['mode'] == 'train':
        wandb = None
        if args['track']:
            wandb = init_wandb(args, id=args['exp_id'])
        train(args, make_env, policy_cls, rnn_cls, wandb=wandb)
    elif args['mode'] in ('eval', 'evaluate'):
        clean_pufferl.rollout(
            make_env,
            args.env,
            policy_cls=policy_cls,
            rnn_cls=rnn_cls,
            agent_creator=make_policy,
            agent_kwargs=args,
            model_path=args.eval_model_path,
            render_mode=args.render_mode,
            device=args.train.device
        )
    elif args['mode'] == 'sweep':
        args['track'] = True
        sweep(args, env_module, make_env, policy_cls, rnn_cls)
    elif args['mode'] == 'sweep-carbs':
        from sweep_carbs import sweep_carbs
        sweep_carbs(args, env_module, make_env)
    elif args['mode'] == 'autotune':
        pufferlib.vector.autotune(make_env, batch_size=args.train.env_batch_size)
    elif args['mode'] == 'profile':
        import cProfile
        cProfile.run('train(args, env_module, make_env)', 'stats.profile')
        import pstats
        from pstats import SortKey
        p = pstats.Stats('stats.profile')
        p.sort_stats(SortKey.TIME).print_stats(10)
