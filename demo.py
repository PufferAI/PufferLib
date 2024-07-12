from pdb import set_trace as T
import functools
import argparse
import shutil
import yaml
import uuid
import sys
import os

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.frameworks.cleanrl

from rich_argparse import RichHelpFormatter
from rich.traceback import install
from rich.console import Console

import clean_pufferl


def load_config(parser, config_path='config.yaml'):
    '''Just a fancy config loader. Populates argparse from
    yaml + env/policy fn signatures to give you a nice
    --help menu + some limited validation of the config'''
    args, _ = parser.parse_known_args()
    env_name, pkg_name = args.env, args.pkg

    with open(config_path) as f:
        config = yaml.safe_load(f)
    if 'default' not in config:
        raise ValueError('Deleted default config section?')
    if env_name not in config:
        raise ValueError(f'{env_name} not in config\n'
            'It might be available through a parent package, e.g.\n'
            '--config atari --env MontezumasRevengeNoFrameskip-v4.')

    default = config['default']
    env_config = config[env_name or pkg_name]
    pkg_name = pkg_name or env_config.get('package', env_name)
    pkg_config = config[pkg_name]
    # TODO: Check if actually installed
    env_module = pufferlib.utils.install_and_import(
        f'pufferlib.environments.{pkg_name}')
    make_name = env_config.get('env_name', None)
    make_env_args = [make_name] if make_name else []
    make_env = env_module.env_creator(*make_env_args)
    make_env_args = pufferlib.utils.get_init_args(make_env)

    if 'policy_name' in env_config:
        config['policy_name'] = env_config['policy_name']
    elif 'policy_name' in pkg_config:
        config['policy_name'] = pkg_config['policy_name']
    else:
        config['policy_name'] = default['policy_name']

    policy_name = config['policy_name']
    policy_cls = getattr(env_module.torch, policy_name)
    policy_args = pufferlib.utils.get_init_args(policy_cls)

    if 'rnn_name' in env_config:
        config['rnn_name'] = env_config['rnn_name']
    elif 'rnn_name' in pkg_config:
        config['rnn_name'] = pkg_config['rnn_name']
    else:
        config['rnn_name'] = default['rnn_name']

    rnn_name = config['rnn_name']
    rnn_cls = getattr(env_module, rnn_name)
    rnn_args = pufferlib.utils.get_init_args(rnn_cls)

    fn_sig = dict(env=make_env_args, policy=policy_args, rnn=rnn_args)
    config = vars(parser.parse_known_args()[0])

    valid_keys = 'env policy rnn train sweep'.split()
    for key in valid_keys:
        fn_subconfig = fn_sig.get(key, {})
        env_subconfig = env_config.get(key, {})
        pkg_subconfig = pkg_config.get(key, {})
        # Priority env->pkg->default->fn config
        config[key] = {**fn_subconfig, **default[key],
            **pkg_subconfig, **env_subconfig}

    for name in valid_keys:
        sub_config = config[name]
        for key, value in sub_config.items():
            data_key = f'{name}.{key}'
            cli_key = f'--{data_key}'.replace('_', '-')
            if isinstance(value, bool) and value is False:
                parser.add_argument(cli_key, default=value, action='store_true')
            elif isinstance(value, bool) and value is True:
                data_key = f'{name}.no_{key}'
                cli_key = f'--{data_key}'.replace('_', '-')
                parser.add_argument(cli_key, default=value, action='store_false')
            else:
                parser.add_argument(cli_key, default=value, type=type(value))

            config[name][key] = getattr(parser.parse_known_args()[0], data_key)
        config[name] = pufferlib.namespace(**config[name])

    pufferlib.utils.validate_args(make_env.func if isinstance(make_env, functools.partial) else make_env, config['env'])
    pufferlib.utils.validate_args(policy_cls, config['policy'])

    parser.add_argument('--policy-name', type=str, default=None, help='Policy name to use')
    config['policy_name'] = parser.parse_known_args()[0].policy_name
    parser.add_argument('--rnn-name', type=str, default=None, help='RNN name to use')
    config['rnn_name'] = parser.parse_known_args()[0].rnn_name

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.parse_args()
    wandb_name = make_name or env_name
    config['env_name'] = env_name
    config['exp_id'] = args.exp_id or args.env + '-' + str(uuid.uuid4())[:8]
    return (wandb_name, pkg_name, pufferlib.namespace(**config),
        env_module, make_env, policy_cls, rnn_cls)
   
def make_policy(env, policy_cls, rnn_cls, args):
    policy = policy_cls(env, **args.policy)
    if rnn_cls is not None:
        policy = rnn_cls(env, policy, **args.rnn)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(args.train.device)

def init_wandb(args, name, id=None, resume=True):
    #os.environ["WANDB_SILENT"] = "true"
    import wandb
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config={
            'cleanrl': dict(args.train),
            'env': dict(args.env),
            'policy': dict(args.policy),
            #'recurrent': args.recurrent,
        },
        name=name,
        monitor_gym=True,
        save_code=True,
        resume=resume,
    )
    return wandb

from math import log, ceil, floor
def closest_power(x):
    possible_results = floor(log(x, 2)), ceil(log(x, 2))
    return int(2**min(possible_results, key= lambda z: abs(x-2**z)))

def sweep_carbs(args, wandb_name, env_module, make_env):
    import wandb
    sweep_id = wandb.sweep(
        sweep=dict(args.sweep),
        project="carbs",
    )
    target_metric = args.sweep['metric']['name'].split('/')[-1]
    wandb_train_params = args.sweep.parameters['train']['parameters']
    wandb_env_params = args.sweep.parameters['env']['parameters']
    wandb_policy_params = args.sweep.parameters['policy']['parameters']

    import numpy as np
    from loguru import logger

    from carbs import CARBS
    from carbs import CARBSParams
    from carbs import LinearSpace
    from carbs import LogSpace
    from carbs import LogitSpace
    from carbs import ObservationInParam
    from carbs import ParamDictType
    from carbs import Param

    logger.remove()
    logger.add(sys.stdout, level="DEBUG", format="{message}")

    def carbs_param(name, space, wandb_params, min=None, max=None,
            search_center=None, is_integer=False, rounding_factor=1):
        wandb_param = wandb_params[name]
        if min is None:
            min = float(wandb_param['min'])
        if max is None:
            max = float(wandb_param['max'])

        if space == 'log':
            Space = LogSpace
            if search_center is None:
                search_center = 2**(np.log2(min) + np.log2(max)/2)
        elif space == 'linear':
            Space = LinearSpace
            if search_center is None:
                search_center = (min + max)/2
        elif space == 'logit':
            Space = LogitSpace
            assert min == 0
            assert max == 1
            assert search_center is not None
        else:
            raise ValueError(f'Invalid CARBS space: {space} (log/linear)')

        return Param(
            name=name,
            space=Space(
                min=min,
                max=max,
                is_integer=is_integer,
                rounding_factor=rounding_factor
            ),
            search_center=search_center,
        )

    # Must be hardcoded and match wandb sweep space for now
    param_spaces = [
        carbs_param('cnn_channels', 'linear', wandb_policy_params, search_center=32, is_integer=True),
        carbs_param('hidden_size', 'linear', wandb_policy_params, search_center=128, is_integer=True),
        #carbs_param('vision', 'linear', search_center=5, is_integer=True),
        #carbs_param('total_timesteps', 'log', wandb_train_params, search_center=1_000_000_000, is_integer=True),
        carbs_param('learning_rate', 'log', wandb_train_params, search_center=9e-4),
        carbs_param('gamma', 'logit', wandb_train_params, search_center=0.99),
        carbs_param('gae_lambda', 'logit', wandb_train_params, search_center=0.90),
        carbs_param('update_epochs', 'linear', wandb_train_params, search_center=1, is_integer=True),
        carbs_param('clip_coef', 'logit', wandb_train_params, search_center=0.1),
        carbs_param('vf_coef', 'logit', wandb_train_params, search_center=0.5),
        carbs_param('vf_clip_coef', 'logit', wandb_train_params, search_center=0.1),
        carbs_param('max_grad_norm', 'linear', wandb_train_params, search_center=0.5),
        carbs_param('ent_coef', 'log', wandb_train_params, search_center=0.07),
        #carbs_param('env_batch_size', 'linear', search_center=384,
        #    is_integer=True, rounding_factor=24),
        carbs_param('batch_size', 'log', wandb_train_params, search_center=262144, is_integer=True),
        carbs_param('minibatch_size', 'log', wandb_train_params, search_center=4096, is_integer=True),
        carbs_param('bptt_horizon', 'log', wandb_train_params, search_center=16, is_integer=True),
    ]

    carbs_params = CARBSParams(
        better_direction_sign=1,
        is_wandb_logging_enabled=False,
        resample_frequency=0,
    )
    carbs = CARBS(carbs_params, param_spaces)

    def main():
            args.exp_name = init_wandb(args, wandb_name, id=args.exp_id)
            orig_suggestion = carbs.suggest().suggestion
            suggestion = orig_suggestion.copy()
            print('Suggestion:', suggestion)
            cnn_channels = suggestion.pop('cnn_channels')
            hidden_size = suggestion.pop('hidden_size')
            #vision = suggestion.pop('vision')
            #wandb.config.env['vision'] = vision
            wandb.config.policy['cnn_channels'] = cnn_channels
            wandb.config.policy['hidden_size'] = hidden_size
            wandb.config.train.update(suggestion)
            wandb.config.train['batch_size'] = closest_power(
                suggestion['batch_size'])
            wandb.config.train['minibatch_size'] = closest_power(
                suggestion['minibatch_size'])
            wandb.config.train['bptt_horizon'] = closest_power(
                suggestion['bptt_horizon'])
            #wandb.config.train['num_envs'] = int(
            #    3*suggestion['env_batch_size'])
            args.train.__dict__.update(dict(wandb.config.train))
            #args.env.__dict__['vision'] = vision
            args.policy.__dict__['cnn_channels'] = cnn_channels
            args.policy.__dict__['hidden_size'] = hidden_size
            args.rnn.__dict__['input_size'] = hidden_size
            args.rnn.__dict__['hidden_size'] = hidden_size
            args.track = True
            print(wandb.config.train)
            print(wandb.config.env)
            print(wandb.config.policy)
            try:
                stats, profile = train(args, env_module, make_env)
            except Exception as e:
                is_failure = True
                import traceback
                traceback.print_exc()
            else:
                observed_value = stats[target_metric]
                uptime = profile.uptime

                with open('hypers.txt', 'a') as f:
                    f.write(f'Train: {args.train.__dict__}\n')
                    f.write(f'Env: {args.env.__dict__}\n')
                    f.write(f'Policy: {args.policy.__dict__}\n')
                    f.write(f'RNN: {args.rnn.__dict__}\n')
                    f.write(f'Uptime: {uptime}\n')
                    f.write(f'Value: {observed_value}\n')

                obs_out = carbs.observe(
                    ObservationInParam(
                        input=orig_suggestion,
                        output=observed_value,
                        cost=uptime,
                    )
                )

    '''
    def main():
            args.exp_name = init_wandb(args, wandb_name, id=args.exp_id)
            suggestion = carbs.suggest().suggestion
            print('Suggestion:', suggestion)
            cnn_channels = suggestion.pop('cnn_channels')
            hidden_size = suggestion.pop('hidden_size')
            #vision = suggestion.pop('vision')
            #wandb.config.env['vision'] = vision
            args.train.__dict__.update(suggestion)
            args.train.__dict__['batch_size'] = closest_power(
                args.train.batch_size)
            args.train.__dict__['minibatch_size'] = closest_power(
                args.train.minibatch_size)
            args.train.__dict__['bptt_horizon'] = closest_power(
                args.train.bptt_horizon)
            args.policy.__dict__.update(suggestion)
            args.policy.__dict__['cnn_channels'] = cnn_channels
            args.policy.__dict__['hidden_size'] = hidden_size
            args.rnn.__dict__['input_size'] = hidden_size
            args.rnn.__dict__['hidden_size'] = hidden_size
            args.track = True
            try:
                stats, profile = train(args, env_module, make_env)
            except Exception as e:
                is_failure = True
                import traceback
                traceback.print_exc()
            else:
                observed_value = stats[target_metric]
                uptime = profile.uptime

                obs_out = carbs.observe(
                    ObservationInParam(
                        input=suggestion,
                        output=observed_value,
                        cost=uptime,
                    )
                )
    main()
    exit(0)
    '''
    wandb.agent(sweep_id, main, count=500)

def sweep(args, wandb_name, env_module, make_env, policy_cls, rnn_cls):
    import wandb
    sweep_id = wandb.sweep(
        sweep=dict(args.sweep),
        project="pufferlib",
    )

    def main():
        try:
            args.exp_name = init_wandb(args, wandb_name, id=args.exp_id)
            # TODO: Add update method to namespace
            print(wandb.config.train)
            args.train.__dict__.update(dict(wandb.config.train))
            args.track = True
            train(args, env_module, make_env, policy_cls, rnn_cls)
        except Exception as e:
            import traceback
            traceback.print_exc()

    wandb.agent(sweep_id, main, count=100)

def train(args, env_module, make_env, policy_cls, rnn_cls):
    args.wandb = None
    args.train.exp_id = args.exp_id
    if args.track:
        args.wandb = init_wandb(args, wandb_name, id=args.exp_id)

    vec = args.vec
    if vec == 'serial':
        vec = pufferlib.vector.Serial
    elif vec == 'multiprocessing':
        vec = pufferlib.vector.Multiprocessing
    elif vec == 'ray':
        vec = pufferlib.vector.Ray
    else:
        raise ValueError(f'Invalid --vector (serial/multiprocessing/ray).')

    vecenv = pufferlib.vector.make(
        make_env,
        env_kwargs=args.env,
        num_envs=args.train.num_envs,
        num_workers=args.train.num_workers,
        batch_size=args.train.env_batch_size,
        zero_copy=args.train.zero_copy,
        backend=vec,
    )
    policy = make_policy(vecenv.driver_env, policy_cls, rnn_cls, args)
    train_config = args.train 
    train_config.track = args.track
    train_config.device = args.train.device
    train_config.env = args.env_name

    if args.backend == 'clean_pufferl':
        data = clean_pufferl.create(train_config, vecenv, policy, wandb=args.wandb)

        while data.global_step < args.train.total_timesteps:
            try:
                clean_pufferl.evaluate(data)
                clean_pufferl.train(data)
            except KeyboardInterrupt:
                clean_pufferl.close(data)
                os._exit(0)
            except Exception:
                Console().print_exception()
                # TODO: breaks sweeps? But needed for training to print cleanly
                os._exit(0)

        stats, _ = clean_pufferl.evaluate(data)
        profile = data.profile
        clean_pufferl.close(data)
        return stats, profile

    elif args.backend == 'sb3':
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.env_util import make_vec_env
        from sb3_contrib import RecurrentPPO

        envs = make_vec_env(lambda: make_env(**args.env),
            n_envs=args.train.num_envs, seed=args.train.seed, vec_env_cls=DummyVecEnv)

        model = RecurrentPPO("CnnLstmPolicy", envs, verbose=1,
            n_steps=args.train.batch_rows*args.train.bptt_horizon,
            batch_size=args.train.batch_size, n_epochs=args.train.update_epochs,
            gamma=args.train.gamma
        )

        model.learn(total_timesteps=args.train.total_timesteps)

if __name__ == '__main__':
    install(show_locals=False) # Rich tracebacks
    # TODO: Add check against old args like --config to demo
    parser = argparse.ArgumentParser(
            description=f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]'
        ' demo options. Shows valid args for your env and policy',
        formatter_class=RichHelpFormatter, add_help=False)
    assert 'config' not in sys.argv, '--config deprecated. Use --env'
    parser.add_argument('--env', '--environment', type=str,
        default='squared', help='Name of specific environment to run')
    parser.add_argument('--pkg', '--package', type=str, default=None, help='Configuration in config.yaml to use')
    parser.add_argument('--backend', type=str, default='clean_pufferl', help='Train backend (clean_pufferl, sb3)')
    parser.add_argument('--mode', type=str, default='train', choices='train eval evaluate sweep sweep-carbs autotune baseline profile'.split())
    parser.add_argument('--eval-model-path', type=str, default=None, help='Path to model to evaluate')
    parser.add_argument('--baseline', action='store_true', help='Baseline run')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices='auto human ansi rgb_array None'.split(),
        help='Disable render during evaluate')
    parser.add_argument('--vec', '--vector', '--vectorization', type=str,
        default='serial', choices='serial multiprocessing ray'.split())
    parser.add_argument('--exp-id', '--exp-name', type=str, default=None, help="Resume from experiment")
    parser.add_argument('--wandb-entity', type=str, default='jsuarez', help='WandB entity')
    parser.add_argument('--wandb-project', type=str, default='pufferlib', help='WandB project')
    parser.add_argument('--wandb-group', type=str, default='debug', help='WandB group')
    parser.add_argument('--track', action='store_true', help='Track on WandB')
    wandb_name, pkg, args, env_module, make_env, policy_cls, rnn_cls = load_config(parser)

    if args.baseline:
        assert args.mode in ('train', 'eval', 'evaluate')
        args.track = True
        version = '.'.join(pufferlib.__version__.split('.')[:2])
        args.exp_id = f'puf-{version}-{args.env_name}'
        args.wandb_group = f'puf-{version}-baseline'
        shutil.rmtree(f'experiments/{args.exp_id}', ignore_errors=True)
        run = init_wandb(args, args.exp_id, resume=False)
        if args.mode in ('eval', 'evaluate'):
            model_name = f'puf-{version}-{args.env_name}_model:latest'
            artifact = run.use_artifact(model_name)
            data_dir = artifact.download()
            model_file = max(os.listdir(data_dir))
            args.eval_model_path = os.path.join(data_dir, model_file)

    if args.mode == 'train':
        train(args, env_module, make_env, policy_cls, rnn_cls)
    elif args.mode in ('eval', 'evaluate'):
        try:
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
        except KeyboardInterrupt:
            os._exit(0)
    elif args.mode == 'sweep':
        sweep(args, wandb_name, env_module, make_env, policy_cls, rnn_cls)
    elif args.mode == 'sweep-carbs':
        sweep_carbs(args, wandb_name, env_module, make_env)
    elif args.mode == 'autotune':
        pufferlib.vector.autotune(make_env, batch_size=args.train.env_batch_size)
    elif args.mode == 'profile':
        import cProfile
        cProfile.run('train(args, env_module, make_env)', 'stats.profile')
        import pstats
        from pstats import SortKey
        p = pstats.Stats('stats.profile')
        p.sort_stats(SortKey.TIME).print_stats(10)
