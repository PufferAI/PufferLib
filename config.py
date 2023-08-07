from pdb import set_trace as T

import pufferlib.vectorization

import torch
from types import SimpleNamespace

def default_config():
    '''Returns a namespace of kwargs'''
    framework = {
        'emulate_const_horizon': 1024,
        'vec_backend': pufferlib.vectorization.Serial,
        'total_timesteps': 10_000_000,
        'learning_rate': 2.5e-4,
        'num_cores': 4,
        'num_buffers': 2,
        'num_envs': 4,
        'batch_size': 1024,
        'batch_rows': 256,
        'bptt_horizon': 1,
        'seed': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'pool_rank_interval': 1,
        'pool_update_policy_interval': 1,
        'pool_add_policy_interval': 1,
    }

    policy = {
        'input_size': 128,
        'hidden_size': 128,
    }

    recurrent = {
        'input_size': 128,
        'hidden_size': 128,
        'num_layers': 0,
    }

    return SimpleNamespace(
        framework=framework,
        policy=policy,
        recurrent=recurrent,
    )

def all():
    '''All tested environments and platforms'''
    return {
        #'atari': atari,
        #'avalon': avalon,
        #'box2d': box2d,
        #'butterfly': butterfly,
        #'crafter': crafter,
        #'dm_control': dm_control,
        #'dm_lab': dm_lab,
        #'griddly': griddly,
        #'magent': magent,
        #'microrts': microrts,
        #'minerl': minerl,
        'nethack': nethack,
        'nmmo': nmmo,
        #'procgen': procgen,
        #'smac': smac,
    }

def atari(framestack=1):
    import pufferlib.registry.atari
    pufferlib.utils.install_requirements('atari')
    config = default_config()
    config.env_creators = {
        pufferlib.registry.atari.make_env: {
            'name': name,
            'framestack': framestack,
        } for name in [
            'BreakoutNoFrameskip-v4',
            'PongNoFrameskip-v4',
        ]
    }
    config.policy_cls = pufferlib.registry.atari.Policy
    config.policy.update({
        'input_size': 512,
        'hidden_size': 128,
        'output_size': 128,
        'framestack': 1,
        'flat_size': 64*7*7,
    })
    config.recurrent.update({
        'input_size': 128,
        'hidden_size': 128,
        'num_layers': 0,
    })
    return config

def avalon():
    import pufferlib.registry.avalon
    pufferlib.utils.install_requirements('avalon')
    config = default_config()
    config.env_creators = {
        pufferlib.registry.avalon.make_env: {}
    }
    config.policy_cls = pufferlib.registry.avalon.Policy
    config.policy.update({
        'input_size': 512,
        'hidden_size': 128,
        'output_size': 128,
        'framestack': 1,
        'flat_size': 64*7*7,
    })
    config.recurrent.update({
        'input_size': 128,
        'hidden_size': 128,
        'num_layers': 0,
    })
    return config

def box2d():
    import pufferlib.registry.box2d
    pufferlib.utils.install_requirements('box2d')
    config = default_config()
    config.env_creators = {
        pufferlib.registry.box2d.make_env: {}
    }
    config.policy_cls = pufferlib.registry.box2d.Policy
    return config

def butterfly():
    import pufferlib.registry.butterfly
    pufferlib.utils.install_requirements('butterfly')
    config = default_config()
    config.env_creators = {
        pufferlib.registry.butterfly.make_cooperative_pong_v5_binding: {}
    }
    config.policy_cls = pufferlib.registry.butterfly.Policy
    return config

def crafter():
    import pufferlib.registry.crafter
    pufferlib.utils.install_requirements('crafter')
    config = default_config()
    config.env_creators = {
        pufferlib.registry.crafter.make_env: {}
    }
    config.policy_cls = pufferlib.registry.crafter.Policy
    config.policy.update({
        'input_size': 512,
        'hidden_size': 128,
        'framestack': 3, # Framestack 3 is a hack for RGB
        'flat_size': 64*4*4,
    })
    return config

def dm_control():
    import pufferlib.registry.dmc
    pufferlib.utils.install_requirements('dm-control')
    config = default_config()
    config.env_creators = {
        pufferlib.registry.dmc.make_env: {}
    }
    config.policy_cls = pufferlib.registry.dmc.Policy
    config.policy.update({
        'input_size': 512,
        'hidden_size': 128,
        'framestack': 3, # Framestack 3 is a hack for RGB
        'flat_size': 64*4*4,
    })
    return config

def dm_lab():
    import pufferlib.registry.dm_lab
    pufferlib.utils.install_requirements('dm-lab')
    config = default_config()
    config.env_creators = {
        pufferlib.registry.dm_lab.make_env: {}
    }
    config.policy_cls = pufferlib.registry.dm_lab.Policy
    config.policy.update({
        'input_size': 512,
        'hidden_size': 128,
        'framestack': 3, # Framestack 3 is a hack for RGB
        'flat_size': 64*4*4,
    })
    return config

def griddly():
    import pufferlib.registry.griddly
    pufferlib.utils.install_requirements('griddly')
    config = default_config()
    config.env_creators = {
        pufferlib.registry.griddly.make_spider_v0_env: {}
    }
    config.policy_cls = pufferlib.registry.griddly.Policy
    return config

def magent():
    import pufferlib.registry.magent
    pufferlib.utils.install_requirements('magent')
    config = default_config()
    config.env_creators = {
        pufferlib.registry.magent.make_battle_v4_env: {}
    }
    config.policy_cls = pufferlib.registry.magent.Policy
    config.policy.update({
        'input_size': 512,
        'hidden_size': 128,
        'output_size': 128,
        'framestack': 5, # Framestack 5 is a hack for obs channels
        'flat_size': 64*4*4,
    })
    return config

def microrts():
    import pufferlib.registry.microrts
    pufferlib.utils.install_requirements('gym-microrts')
    config = default_config()
    config.env_creators = {
        pufferlib.registry.microrts.make_env: {}
    }
    config.policy_cls = pufferlib.registry.microrts.Policy
    return config

def minerl():
    import pufferlib.registry.minecraft
    pufferlib.utils.install_requirements('minerl')
    config = default_config()
    config.env_creators = {
        pufferlib.registry.minecraft.make_env: {}
    }
    config.policy_cls = pufferlib.registry.minecraft.Policy
    return config

def nethack():
    import pufferlib.registry.nethack
    pufferlib.utils.install_requirements('nethack')
    config = default_config()
    config.env_creators = {
        pufferlib.registry.nethack.make_env: {}
    }
    config.policy_cls = pufferlib.registry.nethack.Policy
    config.policy.update({
        'embedding_dim': 32,
        'crop_dim': 9,
        'num_layers': 5,
    })
    return config

def nmmo():
    import pufferlib.registry.nmmo
    pufferlib.utils.install_requirements('nmmo')
    config = default_config()
    config.env_creators = {
        pufferlib.registry.nmmo.make_env: {}
    }
    config.policy_cls = pufferlib.registry.nmmo.Policy
    config.policy.update({
        'batch_size': 2**14,
        'batch_rows': 128,
    })
    return config

def procgen():
    import pufferlib.registry.procgen
    pufferlib.utils.install_requirements('procgen')
    config = default_config()
    config.env_creators = {
        pufferlib.registry.procgen.make_env: {
            'name': 'coinrun'
        }
    }
    config.policy_cls = pufferlib.registry.procgen.Policy
    return config

def smac():
    import pufferlib.registry.smac
    pufferlib.utils.install_requirements('smac')
    config = default_config()
    config.env_creators = {
        pufferlib.registry.smac.make_env: {}
    }
    config.policy_cls = pufferlib.registry.smac.Policy
    config.policy.update({
        'embedding_dim': 32,
        'crop_dim': 9,
        'num_layers': 5,
    })
    return config

# Possible stuff to add support:
# Deep RTS
# https://github.com/kimbring2/MOBA_RL
# Serpent AI