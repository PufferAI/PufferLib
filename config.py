from pdb import set_trace as T

from dataclasses import dataclass, fields

import pufferlib.vectorization
import pufferlib.pytorch

import torch

def struct(frozen=True, **kwargs):
    '''A default frozen dataclass with a .dict() method'''
    def wrapper(cls):
        cls = dataclass(cls, **kwargs)
        cls.dict = lambda self: {field.name: getattr(self, field.name) for field in fields(cls)}
        return cls
    return wrapper

def struct(cls_or_kwargs=None, **kwargs):
    # If called with arguments, return a decorator function
    if cls_or_kwargs is None or isinstance(cls_or_kwargs, dict):
        kwargs.update(cls_or_kwargs or {})
        return lambda cls: struct(cls, **kwargs)
    
    # If called without arguments, modify the class and return it
    cls = cls_or_kwargs
    cls = dataclass(cls, **kwargs)
    cls.dict = lambda self: {field.name: getattr(self, field.name) for field in fields(cls)}
    return cls

@struct()
class CleanRLInit:
    vectorization: ... = pufferlib.vectorization.Multiprocessing
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    num_cores: int = 4
    num_buffers: int = 1
    num_envs: int = 8
    batch_size: int = 1024
    seed: int = 1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb_entity: str = None #'jsuarez' 
    wandb_project: str = 'pufferlib'

@struct
class CleanRLTrain:
    batch_rows: int = 32
    bptt_horizon: int = 8

@struct
class Policy:
    pass

@struct
class Recurrent:
    input_size = 128
    hidden_size = 128
    num_layers = 1

@struct
class Config:
    cleanrl_init: ... = CleanRLInit()
    cleanrl_train: ... = CleanRLTrain()
    policy_cls: ... = pufferlib.models.Default
    policy_kwargs: ... = None
    recurrent_cls: ... = None
    recurrent_kwargs: ... = None
    env_creators: ... = None

def all():
    '''All tested environments and platforms'''
    return {
        #'atari': atari,
        #'avalon': avalon,
        #'box2d': box2d,
        'butterfly': butterfly,
        'crafter': crafter,
        #'dm_control': dm_control,
        #'dm_lab': dm_lab,
        'griddly': griddly,
        'magent': magent,
        #'microrts': microrts,
        #'minerl': minerl,
        'nethack': nethack,
        'nmmo': nmmo,
        'procgen': procgen,
        #'smac': smac,
    }

def atari(framestack=4):
    import pufferlib.registry.atari
    pufferlib.utils.install_requirements('atari')
    return Config(
        env_creators = {
            pufferlib.registry.atari.make_env: {
                'name': name,
                'framestack': framestack,
            } for name in [
                'BreakoutNoFrameskip-v4',
                #'PongNoFrameskip-v4',
            ]
        },
        policy_cls = pufferlib.registry.atari.Policy,
        policy_kwargs = {
            'input_size': 512,
            'hidden_size': 512,
            'output_size': 512,
            'framestack': framestack,
            'flat_size': 64*7*7,
        },
        recurrent_cls = pufferlib.pytorch.BatchFirstLSTM if framestack == 1 else None,
        recurrent_kwargs = {
            'input_size': 128,
            'hidden_size': 128,
            'num_layers': 1,
        } if framestack == 1 else None,
    )

def avalon():
    import pufferlib.registry.avalon
    pufferlib.utils.install_requirements('avalon')
    return Config(
        env_creators = {
            pufferlib.registry.avalon.make_env: {}
        },
        policy_cls = pufferlib.registry.avalon.Policy,
        policy_kwargs = {
            'input_size': 512,
            'hidden_size': 128,
            'output_size': 128,
            'framestack': 1,
            'flat_size': 64*7*7,
        },
        recurrent_cls = pufferlib.pytorch.BatchFirstLSTM,
        recurrent_kwargs = {
            'input_size': 128,
            'hidden_size': 128,
            'num_layers': 0,
        }
    )

def box2d():
    import pufferlib.registry.box2d
    pufferlib.utils.install_requirements('box2d')
    return Config(
        env_creators = {
            pufferlib.registry.box2d.make_env: {}
        },
        policy_cls = pufferlib.registry.box2d.Policy,
        policy_kwargs = {},
    )

def butterfly():
    import pufferlib.registry.butterfly
    pufferlib.utils.install_requirements('butterfly')
    return Config(
        env_creators = {
            pufferlib.registry.butterfly.make_cooperative_pong_v5: {}
        },
        policy_cls = pufferlib.registry.butterfly.Policy,
        policy_kwargs = {},
    )

def classic_control():
    import pufferlib.registry.classic_control
    return Config(
        env_creators = {
            pufferlib.registry.classic_control.make_cartpole_env: {}
        },
        cleanrl_init = CleanRLInit(
            vectorization=pufferlib.vectorization.Serial,
            total_timesteps=10_000_000,
            num_cores=1,
            num_buffers=1,
            num_envs=16,
            batch_size=1024,
        ),
        policy_cls = pufferlib.registry.classic_control.Policy,
        policy_kwargs = {
            'input_size': 64,
            'hidden_size': 64,
        },
    )

def crafter():
    import pufferlib.registry.crafter
    pufferlib.utils.install_requirements('crafter')
    return Config(
        env_creators = {
            pufferlib.registry.crafter.make_env: {}
        },
        policy_cls = pufferlib.registry.crafter.Policy,
        policy_kwargs = {},
    )

def dm_control():
    import pufferlib.registry.dmc
    pufferlib.utils.install_requirements('dm-control')
    return Config(
        env_creators = {
            pufferlib.registry.dmc.make_env: {}
        },
        policy_cls = pufferlib.registry.dmc.Policy,
        policy_kwargs = {
            'input_size': 512,
            'hidden_size': 128,
            'framestack': 3, # Framestack 3 is a hack for RGB
            'flat_size': 64*4*4,
        },
    )

def dm_lab():
    import pufferlib.registry.dm_lab
    pufferlib.utils.install_requirements('dm-lab')
    return Config(
        env_creators = {
            pufferlib.registry.dm_lab.make_env: {}
        },
        policy_cls = pufferlib.registry.dm_lab.Policy,
        policy_kwargs = {},
    )

def griddly():
    import pufferlib.registry.griddly
    pufferlib.utils.install_requirements('griddly')
    return Config(
        env_creators = {
            pufferlib.registry.griddly.make_spider_v0_env: {}
        },
        policy_cls = pufferlib.registry.griddly.Policy,
        policy_kwargs = {},
    )

def magent():
    import pufferlib.registry.magent
    pufferlib.utils.install_requirements('magent')
    return Config(
        env_creators = {
            pufferlib.registry.magent.make_battle_v4_env: {}
        },
        policy_cls = pufferlib.registry.magent.Policy,
        policy_kwargs = {
            'input_size': 512,
            'hidden_size': 128,
            'output_size': 128,
            'framestack': 5, # Framestack 5 is a hack for obs channels
            'flat_size': 64*4*4,
        },
    )

def microrts():
    import pufferlib.registry.microrts
    pufferlib.utils.install_requirements('gym-microrts')
    return Config(
        env_creators = {
            pufferlib.registry.microrts.make_env: {}
        },
        policy_cls = pufferlib.registry.microrts.Policy,
        policy_kwargs = {},
    )

def minerl():
    import pufferlib.registry.minecraft
    pufferlib.utils.install_requirements('minerl')
    return Config(
        env_creators = {
            pufferlib.registry.minecraft.make_env: {}
        },
        policy_cls = pufferlib.registry.minecraft.Policy,
        policy_kwargs = {},
    )

def nethack():
    import pufferlib.registry.nethack
    pufferlib.utils.install_requirements('nethack')
    return Config(
        env_creators = {
            pufferlib.registry.nethack.make_env: {}
        },
        policy_cls = pufferlib.registry.nethack.Policy,
        policy_kwargs = {
            'embedding_dim': 32,
            'crop_dim': 9,
            'num_layers': 5,
        },
        recurrent_kwargs = {
            'input_size': 512,
            'hidden_size': 512,
        },
    )

def nmmo():
    import pufferlib.registry.nmmo
    pufferlib.utils.install_requirements('nmmo')
    return Config(
        env_creators = {
            pufferlib.registry.nmmo.make_env: {}
        },
        cleanrl_init = CleanRLInit(
            batch_size=2**12,
            num_cores=1,
            num_buffers=1,
            num_envs=1,
        ),
        cleanrl_train = CleanRLTrain(
            batch_rows=128,
        ),
        policy_cls = pufferlib.registry.nmmo.Policy,
        policy_kwargs = {},
        recurrent_cls = pufferlib.pytorch.BatchFirstLSTM,
        recurrent_kwargs = {
            'input_size': 256,
            'hidden_size': 256,
        }
    )

def procgen():
    import pufferlib.registry.procgen
    pufferlib.utils.install_requirements('procgen')
    return Config(
        env_creators = {
            pufferlib.registry.procgen.make_env: {
                'name': 'coinrun'
            }
        },
        policy_cls = pufferlib.registry.procgen.Policy,
        policy_kwargs = {},
    )

def smac():
    import pufferlib.registry.smac
    pufferlib.utils.install_requirements('smac')
    return Config(
        env_creators = {
            pufferlib.registry.smac.make_env: {}
        },
        policy_cls = pufferlib.registry.smac.Policy,
        policy_kwargs = {
            'embedding_dim': 32,
            'crop_dim': 9,
            'num_layers': 5,
        },
    )

# Possible stuff to add support:
# Deep RTS
# https://github.com/kimbring2/MOBA_RL
# Serpent AI
