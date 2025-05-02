import configparser
import argparse
import shutil
import glob
import uuid
import ast
import os
import random
import time

import numpy as np
import torch

import pufferlib
import pufferlib.sweep
import pufferlib.utils
import pufferlib.vector

from rich_argparse import RichHelpFormatter
from rich.console import Console
from rich.traceback import install
install(show_locals=False) # Rich tracebacks

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

import clean_pufferl
 
def init_wandb(args, name, id=None, resume=True, tag=None):
    import wandb
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args['wandb_project'],
        group=args['wandb_group'],
        allow_val_change=True,
        save_code=False,
        resume=resume,
        config=args,
        name=name,
        tags=[tag] if tag is not None else [],
    )
    return wandb

def init_neptune(args, name, id=None, resume=True, tag=None, mode="async"):
    import neptune
    import neptune.exceptions
    try:
        workspace = args['workspace']
        run = neptune.init_run(
                project=f"{workspace['name']}/{workspace['project']}",
                capture_hardware_metrics=False,
                capture_stdout=False,
                capture_stderr=False,
                capture_traceback=False,
                tags=[tag] if tag is not None else [],
                mode=mode,
            )
    except neptune.exceptions.NeptuneConnectionLostException:
        print("couldn't connect to neptune, logging in offline mode")
        return init_neptune(args, name, id, resume, tag, mode="offline")
    return run

def make_policy(env, policy_cls, rnn_cls, args):
    policy = policy_cls(env, **args['policy'],
        #batch_size=args['train']['batch_size'],
        use_p3o=args['train']['use_p3o'],
        p3o_horizon=args['train']['p3o_horizon'],
        use_diayn=args['train']['use_diayn'],
        diayn_skills=args['train']['diayn_archive'],
    )
    args['rnn']['input_size'] = policy.hidden_size
    args['rnn']['hidden_size'] = policy.hidden_size
    if rnn_cls is not None:
        policy = rnn_cls(env, policy, **args['rnn'])

    return policy.to(args['train']['device'])

def sweep(args, env_name, make_env, policy_cls, rnn_cls):
    method = args['sweep']['method']
    if method == 'random':
        sweep = pufferlib.sweep.Random(args['sweep'])
    elif method == 'pareto_genetic':
        sweep = pufferlib.sweep.ParetoGenetic(args['sweep'])
    elif method == 'protein':
        sweep = pufferlib.sweep.Protein(
            args['sweep'],
            resample_frequency=0,
            num_random_samples=50, # Should be number of params
            max_suggestion_cost=args['max_suggestion_cost'],
            min_score = args['sweep']['metric']['min'],
            max_score = args['sweep']['metric']['max'],
        )
    elif method == 'carbs':
        sweep = pufferlib.sweep.Carbs(
            args['sweep'],
            resample_frequency=5,
            num_random_samples=10, # Should be number of params
            max_suggestion_cost=args['max_suggestion_cost'],
        )
    else:
        raise ValueError(f'Invalid sweep method {method} (random/pareto_genetic/protein)')

    target_metric = args['sweep']['metric']['name']
    for i in range(args['max_runs']):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        info = sweep.suggest(args)
        if args['train']['minibatch_size'] >= args['train']['batch_size']:
            sweep.observe(args, 0.0, 0.0)
            continue
        
        scores, costs, timesteps, _, _ = train(args, make_env, policy_cls, rnn_cls, target_metric)

        # Hacky patch to prevent increasing total_timesteps when not swept
        total_timesteps = args['train']['total_timesteps']
        for score, cost, timestep in zip(scores, costs, timesteps):
            args['train']['total_timesteps'] = timestep
            sweep.observe(args, score, cost)

        args['train']['total_timesteps'] = total_timesteps

        print('Score:', score, 'Cost:', cost, 'Timesteps:', timestep)

def train(args, make_env, policy_cls, rnn_cls, target_metric, min_eval_points=100,
        elos={'model_random.pt': 1000}, vecenv=None, wandb=None, neptune=None):
    if args['vec'] == 'serial':
        vec = pufferlib.vector.Serial
    elif args['vec'] == 'multiprocessing':
        vec = pufferlib.vector.Multiprocessing
    elif args['vec'] == 'ray':
        vec = pufferlib.vector.Ray
    elif args['vec'] == 'native':
        vec = pufferlib.environment.PufferEnv
    else:
        raise ValueError(f'Invalid --vec (serial/multiprocessing/ray/native).')

    env_name = args['env_name']
    if vecenv is None:
        vecenv = pufferlib.vector.make(
            make_env,
            env_kwargs=args['env'],
            num_envs=args['train']['num_envs'],
            num_workers=args['train']['num_workers'],
            batch_size=args['train']['env_batch_size'],
            zero_copy=args['train']['zero_copy'],
            overwork=args['vec_overwork'],
            seed=args['train']['seed'],
            backend=vec,
        )

    policy = make_policy(vecenv.driver_env, policy_cls, rnn_cls, args)

    if args['ddp']:
        from torch.nn.parallel import DistributedDataParallel as DDP
        orig_policy = policy
        policy = DDP(policy, device_ids=[args['rank']])
        # TODO: Test this? isinstance?
        if hasattr(orig_policy, 'lstm'):
            policy.lstm = orig_policy.lstm

    neptune = None
    wandb = None
    if args['neptune']:
        neptune = init_neptune(args, env_name, id=args['exp_id'], tag=args['tag'])
        for k, v in pufferlib.utils.unroll_nested_dict(args):
            neptune[k].append(v)
    elif args['wandb']:
        wandb = init_wandb(args, env_name, id=args['exp_id'], tag=args['tag'])

    train_config = pufferlib.namespace(**args['train'], env=env_name,
        exp_id=args['exp_id'] or env_name + '-' + str(uuid.uuid4())[:8])
    data = clean_pufferl.create(train_config, vecenv, policy, wandb=wandb, neptune=neptune)

    timesteps = []
    scores = []
    costs = []
    target_key = f'environment/{target_metric}'

    vecenv.async_reset(train_config.seed)
    while data.global_step < train_config.total_timesteps:
        clean_pufferl.evaluate(data)
        logs = clean_pufferl.train(data)
        if logs is not None and target_key in logs:
            timesteps.append(logs['agent_steps'])
            scores.append(logs[target_key])
            #costs.append(data.profile.uptime)

    steps_evaluated = 0
    cost = time.time() - data.start_time
    batch_size = args['train']['batch_size']
    while len(data.stats[target_metric]) < min_eval_points:
        stats, _ = clean_pufferl.evaluate(data)
        steps_evaluated += batch_size

    clean_pufferl.mean_and_log(data)
    score = stats[target_metric]
    print(f'Evaluated {steps_evaluated} steps. Score: {score}')

    scores.append(score)
    costs.append(cost)
    timesteps.append(data.global_step)

    def downsample_linear(arr, m):
        n = len(arr)
        x_old = np.linspace(0, 1, n)  # Original indices normalized
        x_new = np.linspace(0, 1, m)  # New indices normalized
        return np.interp(x_new, x_old, arr)
     
    scores = downsample_linear(scores, 10)
    costs = downsample_linear(costs, 10)
    timesteps = downsample_linear(timesteps, 10)

    if args['neptune']:
        neptune['score'].append(score)
        neptune['cost'].append(cost)
    elif args['wandb']:
        wandb.log({'score': score, 'cost': cost})

    clean_pufferl.close(data)
    return scores, costs, timesteps, elos, vecenv

def train_ddp(rank, world_size, args, make_env, policy_cls, rnn_cls, target_metric):
    import torch.distributed as dist
    args['rank'] = rank
    args['train']['device'] = f'cuda:{rank}'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    train(args, make_env, policy_cls, rnn_cls, target_metric)
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]'
        ' demo options. Shows valid args for your env and policy',
        formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--env', '--environment', type=str,
        default='puffer_squared', help='Name of specific environment to run')
    parser.add_argument('--mode', type=str, default='train',
        choices='train eval evaluate sweep autotune profile'.split())
    parser.add_argument('--vec-overwork', action='store_true',
        help='Allow vectorization to use >1 worker/core. Not recommended.')
    parser.add_argument('--eval-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint')
    parser.add_argument('--baseline', action='store_true',
        help='Load pretrained model from WandB if available')
    parser.add_argument('--ddp', action='store_true', help='Distributed data parallel')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--exp-id', '--exp-name', type=str,
        default=None, help='Resume from experiment')
    parser.add_argument('--data-path', type=str, default=None,
        help='Used for testing hparam algorithms')
    parser.add_argument('--track', action='store_true', help='Track on WandB')
    parser.add_argument('--max-runs', type=int, default=200, help='Max number of sweep runs')
    parser.add_argument('--wandb-project', type=str, default='pufferlib')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    parser.add_argument('--wandb', action='store_true', help='Track on WandB')
    parser.add_argument('--neptune', action='store_true', help='Track on Neptune')
    #parser.add_argument('--wandb-project', type=str, default='pufferlib')
    #parser.add_argument('--wandb-group', type=str, default='debug')
    args = parser.parse_known_args()[0]

    file_paths = glob.glob('config/**/*.ini', recursive=True)
    for path in file_paths:
        p = configparser.ConfigParser()
        p.read('config/default.ini')

        subconfig = os.path.join(*path.split('/')[:-1] + ['default.ini'])
        if subconfig in file_paths:
            p.read(subconfig)

        p.read(path)
        if args.env in p['base']['env_name'].split():
            break
    else:
        raise Exception('No config for env_name {}'.format(args.env))

    for section in p.sections():
        for key in p[section]:
            if section == 'base':
                argparse_key = f'--{key}'.replace('_', '-')
            else:
                argparse_key = f'--{section}.{key}'.replace('_', '-')
            parser.add_argument(argparse_key, default=p[section][key])

    # Late add help so you get a dynamic menu based on the env
    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    parsed = parser.parse_args().__dict__
    args = {'env': {}, 'policy': {}, 'rnn': {}}
    env_name = parsed.pop('env')
    for key, value in parsed.items():
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

    package = args['package']
    module_name = f'pufferlib.environments.{package}'
    if package == 'ocean':
        module_name = 'pufferlib.ocean'

    import importlib
    env_module = importlib.import_module(module_name)

    make_env = env_module.env_creator(env_name)
    policy_cls = getattr(env_module.torch, args['policy_name'])
    
    rnn_name = args['rnn_name']
    rnn_cls = None
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args['rnn_name'])

    if args['baseline']:
        assert args['mode'] in ('train', 'eval', 'evaluate')
        args['track'] = True
        version = '.'.join(pufferlib.__version__.split('.')[:2])
        args['exp_id'] = f'puf-{version}-{env_name}'
        args['wandb_group'] = f'puf-{version}-baseline'
        shutil.rmtree(f'experiments/{args["exp_id"]}', ignore_errors=True)
        run = init_wandb(args, args['exp_id'], resume=False)
        if args['mode'] in ('eval', 'evaluate'):
            model_name = f'puf-{version}-{env_name}_model:latest'
            artifact = run.use_artifact(model_name)
            data_dir = artifact.download()
            model_file = max(os.listdir(data_dir))
            args['eval_model_path'] = os.path.join(data_dir, model_file)
    if args['mode'] == 'train' and args['ddp']:
        import torch.multiprocessing as mp
        world_size = 1
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        target_metric = args['sweep']['metric']['name']
        mp.spawn(train_ddp,
            args=(world_size, args, make_env, policy_cls, rnn_cls, target_metric),
            nprocs=world_size,
            join=True,
        )
    elif args['mode'] == 'train':
        target_metric = args['sweep']['metric']['name']
        train(args, make_env, policy_cls, rnn_cls, target_metric)
    elif args['mode'] in ('eval', 'evaluate'):
        vec = pufferlib.vector.Serial
        if args['vec'] == 'native': vec = pufferlib.environment.PufferEnv
        clean_pufferl.rollout(
            make_env,
            args['env'],
            policy_cls=policy_cls,
            rnn_cls=rnn_cls,
            agent_creator=make_policy,
            agent_kwargs=args,
            backend=vec,
            model_path=args['eval_model_path'],
            render_mode=args['render_mode'],
            device=args['train']['device'],
        )
    elif args['mode'] == 'sweep':
        assert args['wandb'] or args['neptune'], 'Sweeps require either wandb or neptune'
        sweep(args, env_name, make_env, policy_cls, rnn_cls)
    elif args['mode'] == 'autotune':
        pufferlib.vector.autotune(make_env, batch_size=args['train']['env_batch_size'])
    elif args['mode'] == 'profile':
        import cProfile
        target_metric = args['sweep']['metric']['name']
        cProfile.run('train(args, make_env, policy_cls, rnn_cls, target_metric)', 'stats.profile')
        import pstats
        from pstats import SortKey
        p = pstats.Stats('stats.profile')
        p.sort_stats(SortKey.TIME).print_stats(10)
        breakpoint()
        pass
