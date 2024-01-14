from pdb import set_trace as T
import argparse
import shutil
import sys
import os

import importlib
import inspect
import yaml

import pufferlib
import pufferlib.utils

from clean_pufferl import CleanPuffeRL, rollout, done_training


def load_from_config(env):
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    assert env in config, f'"{env}" not found in config.yaml. Uncommon environments that are part of larger packages may not have their own config. Specify these manually using the parent package, e.g. --config atari --env MontezumasRevengeNoFrameskip-v4.'

    default_keys = 'env train policy recurrent sweep_metadata sweep_metric sweep'.split()
    defaults = {key: config.get(key, {}) for key in default_keys}

    # Package and subpackage (environment) configs
    env_config = config[env]
    pkg = env_config['package']
    pkg_config = config[pkg]

    combined_config = {}
    for key in default_keys:
        env_subconfig = env_config.get(key, {})
        pkg_subconfig = pkg_config.get(key, {})

        # Override first with pkg then with env configs
        combined_config[key] = {**defaults[key], **pkg_subconfig, **env_subconfig}

    return pkg, pufferlib.namespace(**combined_config)
   
def make_policy(env, env_module, args):
    policy = env_module.Policy(env, **args.policy)
    if args.force_recurrence or env_module.Recurrent is not None:
        policy = env_module.Recurrent(env, policy, **args.recurrent)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(args.train.device)

def init_wandb(args, env_module, name=None, resume=True):
    #os.environ["WANDB_SILENT"] = "true"

    import wandb
    return wandb.init(
        id=args.exp_name or wandb.util.generate_id(),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config={
            'cleanrl': args.train,
            'env': args.env,
            'policy': args.policy,
            'recurrent': args.recurrent,
        },
        name=name or args.config,
        monitor_gym=True,
        save_code=True,
        resume=resume,
    )

def sweep(args, env_module, make_env):
    import wandb
    sweep_id = wandb.sweep(sweep=args.sweep, project="pufferlib")

    def main():
        try:
            args.exp_name = init_wandb(args, env_module)
            if hasattr(wandb.config, 'train'):
                # TODO: Add update method to namespace
                print(args.train.__dict__)
                print(wandb.config.train)
                args.train.__dict__.update(dict(wandb.config.train))
            train(args, env_module, make_env)
        except Exception as e:
            import traceback
            traceback.print_exc()

    wandb.agent(sweep_id, main, count=20)

def get_init_args(fn):
    if fn is None:
        return {}

    sig = inspect.signature(fn)
    args = {}
    for name, param in sig.parameters.items():
        if name in ('self', 'env', 'policy'):
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        else:
            args[name] = param.default if param.default is not inspect.Parameter.empty else None
    return args

def train(args, env_module, make_env):
    if args.backend == 'clean_pufferl':
        trainer = CleanPuffeRL(
            config=args.train,
            agent_creator=make_policy,
            agent_kwargs={'env_module': env_module, 'args': args},
            env_creator=make_env,
            env_creator_kwargs=args.env,
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
    parser = argparse.ArgumentParser(description='Parse environment argument', add_help=False)
    parser.add_argument('--backend', type=str, default='clean_pufferl', help='Train backend (clean_pufferl, sb3)')
    parser.add_argument('--config', type=str, default='pokemon_red', help='Configuration in config.yaml to use')
    parser.add_argument('--env', type=str, default=None, help='Name of specific environment to run')
    parser.add_argument('--mode', type=str, default='train', help='train/sweep/evaluate')
    parser.add_argument('--eval-model-path', type=str, default=None, help='Path to model to evaluate')
    parser.add_argument('--baseline', action='store_true', help='Baseline run')
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
    pkg, config = load_from_config(args['config'])

    try:
        env_module = importlib.import_module(f'pufferlib.environments.{pkg}')
    except:
        pufferlib.utils.install_requirements(pkg)
        env_module = importlib.import_module(f'pufferlib.environments.{pkg}')

    # Get the make function for the environment
    env_name = args['env'] or config.env.pop('name')
    make_env = env_module.env_creator(env_name)

    # Update config with environment defaults
    config.env = {**get_init_args(make_env), **config.env}
    config.policy = {**get_init_args(env_module.Policy.__init__), **config.policy}
    config.recurrent = {**get_init_args(env_module.Recurrent.__init__), **config.recurrent}

    # Generate argparse menu from config
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
        args[name] = pufferlib.namespace(**args[name])

    clean_parser.parse_args(sys.argv[1:])
    args = pufferlib.namespace(**args)

    vec = args.vectorization
    if vec == 'serial':
        args.vectorization = pufferlib.vectorization.Serial
    elif vec == 'multiprocessing':
        args.vectorization = pufferlib.vectorization.Multiprocessing
    elif vec == 'ray':
        args.vectorization = pufferlib.vectorization.Ray
    else:
        raise ValueError(f'Invalid --vectorization (serial/multiprocessing/ray).')

    if args.mode == 'sweep':
        args.track = True
    elif args.track:
        args.exp_name = init_wandb(args, env_module).id
    elif args.baseline:
        args.track = True
        args.exp_name = args.config
        args.wandb_group = f'puf-{pufferlib.__version__}-baseline'
        shutil.rmtree(f'experiments/{args.exp_name}', ignore_errors=True)
        run = init_wandb(args, env_module, name=args.exp_name, resume=False)
        if args.mode == 'evaluate':
            model_name = f'puf{pufferlib.__version__}-{args.config}_model:latest'
            artifact = run.use_artifact(model_name)
            data_dir = artifact.download()
            model_file = max(os.listdir(data_dir))
            args.eval_model_path = os.path.join(data_dir, model_file)

    if args.mode == 'train':
        train(args, env_module, make_env)
        exit(0)
    elif args.mode == 'sweep':
        sweep(args, env_module, make_env)
        exit(0)
    elif args.mode == 'evaluate' and pkg != 'pokemon_red':
        rollout(
            make_env,
            args.env,
            agent_creator=make_policy,
            agent_kwargs={'env_module': env_module, 'args': args},
            model_path=args.eval_model_path,
            device=args.train.device
        )
    elif args.mode == 'evaluate' and pkg == 'pokemon_red':
        import pokemon_red_eval
        pokemon_red_eval.rollout(
            make_env,
            args.env,
            agent_creator=make_policy,
            agent_kwargs={'env_module': env_module, 'args': args},
            model_path=args.eval_model_path,
            device=args.train.device,
        )
    elif pkg != 'pokemon_red':
        raise ValueError('Mode must be one of train, sweep, or evaluate')
