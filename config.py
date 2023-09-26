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

def default():
    return CleanRLInit(), CleanRLTrain()

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
    return cleanrl_init, CleanRLTrain()

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
    return cleanrl_init, cleanrl_train

def squared():
    cleanrl_init = CleanRLInit(
        learning_rate=2.5e-2,
    )
    cleanrl_train = CleanRLTrain(
        batch_rows=32,
        bptt_horizon=4,
    )
    return cleanrl_init, cleanrl_train
