from pdb import set_trace as T

import pufferlib.args
import pufferlib.vectorization


@pufferlib.dataclass
class SweepMetadata:
    method: str = 'random'
    name: str = 'sweep'

@pufferlib.dataclass
class SweepMetric:
    goal = 'maximize'
    name = 'episodic_return'

@pufferlib.dataclass
class CleanPuffeRLSweep:
    learning_rate = {
        'distribution': 'log_uniform_values',
        'min': 1e-4,
        'max': 1e-1,
    }
    batch_size = {
        'values': [128, 256, 512, 1024, 2048],
    }
    batch_rows = {
        'values': [16, 32, 64, 128, 256],
    }
    bptt_horizon = {
        'values': [4, 8, 16, 32],
    }

def make_sweep_config(method='random', name='sweep',
        metric=None, cleanrl=None, env=None, policy=None):
    sweep_parameters = {}
    if metric is None:
        sweep_metric = dict(SweepMetric())
    else:
        sweep_metric = dict(metric)

    if cleanrl is not None:
        sweep_parameters['cleanrl'] = {'parameters': dict(cleanrl)}
    if env is not None:
        sweep_parameters['env'] = {'parameters': dict(env)}
    if policy is not None:
        sweep_parameters['policy'] = {'parameters': dict(policy)}
        
    return {
        'method': method,
        'name': name,
        'metric': sweep_metric,
        'parameters': sweep_parameters,
    }
 
def default():
   return pufferlib.args.CleanPuffeRL(), make_sweep_config()

def all():
    '''All tested environments and platforms'''
    return {
        'atari': default,
        #'box2d': default, #Continuous
        'butterfly': default,
        'classic_control': classic_control,
        'crafter': default,
        'squared': squared,
        'dm_control': default,
        'dm_lab': default,
        'griddly': default,
        'magent': default,
        'microrts': default,
        'minerl': default,
        'minigrid': default,
        'minihack': default,
        'nethack': default,
        'nmmo': nmmo,
        'open_spiel': open_spiel,
        'pokemon_red': pokegym,
        'pokemon_red_pip': pokegym,
        'links_awaken': pokegym,
        'procgen': procgen,
        #'smac': default,
        #'stable-retro': default,
    }

def classic_control():
    args = pufferlib.args.CleanPuffeRL(
        vectorization=pufferlib.vectorization.Serial,
        num_envs=16,
    )
    return args, make_sweep_config()

def nmmo():
    args = pufferlib.args.CleanPuffeRL(
        num_envs=64,
        envs_per_batch=24,
        envs_per_worker=1,
        batch_size=2**16,
        batch_rows=128,
    )
    return args, make_sweep_config()

def open_spiel():
    from itertools import chain
    num_opponents = 1
    args = pufferlib.args.CleanPuffeRL(
        pool_kernel = list(chain.from_iterable(
            [[0, i, i, 0] for i in range(1, num_opponents + 1)])),
        num_envs = 32,
        batch_size = 4096,
    )
    sweep_config = make_sweep_config(
            cleanrl=CleanPuffeRLSweep(),
    )
    return args, sweep_config

def pokegym():
    args = pufferlib.args.CleanPuffeRL(
        total_timesteps=100_000_000,
        num_envs=64,
        envs_per_worker=1,
        envpool_batch_size=24,
        update_epochs=3,
        gamma=0.998,
        batch_size=2**15,
        batch_rows=128,
    )
    return args, make_sweep_config()

def procgen():
    # MSRL defaults. Don't forget to uncomment network layer sizes!
    '''
    args = pufferlib.args.CleanPuffeRL(
        total_timesteps=8_000_000,
        learning_rate=6e-4,
        num_cores=4,
        num_envs=64,
        batch_size=2048,
        batch_rows=8,
        bptt_horizon=256,
        gamma=0.995,
        gae_lambda=0.8,
        clip_coef=0.1,
        vf_clip_coef=1.0,
        ent_coef=0.005,
    )
    '''

    # 2020 Competition Defaults from RLlib
    '''
    args = pufferlib.args.CleanPuffeRL(
        total_timesteps=8_000_000,
        learning_rate=5e-4,
        num_cores=1, #4
        num_envs=1,#32,#6,
        batch_size=16384,
        batch_rows=8,
        bptt_horizon=256,
        gamma=0.999,
        update_epochs=3,
        anneal_lr=False,
        clip_coef=0.2,
        vf_clip_coef=0.2,
    )
    '''

    # Experimental defaults from CARBS
    args = pufferlib.args.CleanPuffeRL(
        total_timesteps=8_000_000,
        learning_rate=0.0002691194621325188,
        gamma=0.9961844515798506,
        gae_lambda=0.8730081151095287,
        ent_coef=0.0103205077441882,
        vf_coef=1.9585588391335327,
        clip_coef=0.3075580869152367,
        batch_size=16384, #94208,
        batch_rows=4096,
        bptt_horizon=1,
        update_epochs=2,
    )
    return args, make_sweep_config()

def squared():
    args = pufferlib.args.CleanPuffeRL(
        total_timesteps=30_000,
        learning_rate=0.017,
        num_envs=8,
        batch_rows=32,
        bptt_horizon=4,
    )
    sweep_config = make_sweep_config(
        metric=SweepMetric(name='stats/targets_hit'),
        cleanrl=CleanPuffeRLSweep(),
    )
    return args, sweep_config

def stable_retro():
    # Retro cannot simulate multiple environments per core
    args = pufferlib.args.CleanPuffeRL(
        vectorization=pufferlib.vectorization.Multiprocessing,
        num_envs=1,
    )
    return args, make_sweep_config()
