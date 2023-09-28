from pdb import set_trace as T
import pufferlib.vectorization

import torch


@pufferlib.dataclass
class CleanRLInit:
    vectorization: ... = pufferlib.vectorization.Serial
    #vectorization: ... = pufferlib.vectorization.Multiprocessing
    total_timesteps: int = 30_000 # 10_000_000
    learning_rate: float = 2.5e-4
    num_cores: int = 4
    num_buffers: int = 1
    num_envs: int = 8
    batch_size: int = 1024
    seed: int = 1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@pufferlib.dataclass
class CleanRLTrain:
    batch_rows: int = 32
    bptt_horizon: int = 8

@pufferlib.dataclass
class SweepMetadata:
    method: str = 'random'
    name: str = 'sweep'

@pufferlib.dataclass
class SweepMetric:
    goal = 'maximize'
    name = 'episodic_return'

@pufferlib.dataclass
class CleanRLInitSweep:
    learning_rate = {
        'distribution': 'log_uniform_values',
        'min': 1e-4,
        'max': 1e-1,
    }
    batch_size = {
        'values': [128, 256, 512, 1024, 2048],
    }

@pufferlib.dataclass
class CleanRLTrainSweep:
    batch_rows = {
        'values': [16, 32, 64, 128, 256],
    }
    bptt_horizon = {
        'values': [4, 8, 16, 32],
    }

def make_sweep_config(method='random', name='sweep',
        metric=None, cleanrl_init=None, cleanrl_train=None,
        env=None, policy=None,):
    sweep_parameters = {}
    if metric is None:
        sweep_metric = dict(SweepMetric())
    else:
        sweep_metric = dict(metric)

    if cleanrl_init is not None:
        sweep_parameters['cleanrl_init'] = {'parameters': dict(cleanrl_init)}
    if cleanrl_train is not None:
        sweep_parameters['cleanrl_init'] = {'parameters': dict(cleanrl_init)}
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
   return CleanRLInit(), CleanRLTrain(), make_sweep_config()

def all():
    '''All tested environments and platforms'''
    return {
        'atari': default,
        #'box2d': default,
        'butterfly': default,
        'classic_control': classic_control,
        'crafter': default,
        'squared': squared,
        #'dm_control': default,
        'dm_lab': default,
        'griddly': default,
        'magent': default,
        #'microrts': default,
        #'minerl': default,
        'nethack': default,
        'nmmo': nmmo,
        'procgen': default,
        #'smac': default,
    }

def classic_control():
    cleanrl_init = CleanRLInit(
        vectorization=pufferlib.vectorization.Serial,
        num_cores=1,
        num_buffers=1,
        num_envs=16,
    )
    return cleanrl_init, CleanRLTrain(), make_sweep_config()

def nmmo():
    cleanrl_init = CleanRLInit(
        batch_size=2**12,
        num_cores=1,
        num_buffers=1,
        num_envs=1,
    )
    cleanrl_train = CleanRLTrain(
        batch_rows=128,
    )
    return cleanrl_init, cleanrl_train, make_sweep_config()

def squared():
    cleanrl_init = CleanRLInit(
        learning_rate=0.017,
    )
    cleanrl_train = CleanRLTrain(
        batch_rows=32,
        bptt_horizon=4,
    )
    sweep_config = make_sweep_config(
        metric=SweepMetric(name='stats/targets_hit'),
        cleanrl_init=CleanRLInitSweep(),
    )
    return cleanrl_init, cleanrl_train, sweep_config
