from pdb import set_trace as T

import gym
import numpy as np

import pufferlib
import pufferlib.binding
import pufferlib.utils


def print_setup_error(env: str):
    print(f'{env}: Binding setup failed and will not be available.')

def make_all_atari_bindings():
    '''Make every single Atari binding. Not used for standard tests'''
    try:
        env_ids = [e.id for e in gym.envs.registry.all()]
        with pufferlib.utils.Suppress():
            envs = [gym.make(e) for e in env_ids]
    except:
        print_setup_error('Atari (ale)')
    else:
        return [
            pufferlib.binding.auto(
                env=e,
                env_name=env_id,
            )
            for env_id, e in zip(env_ids, envs)
        ]
  
def make_atari_bindings():
    '''Make DQN paper Atari games'''
    try:
        env_ids = 'BeamRider-v4 Breakout-v4 Enduro-v4 Pong-v4 Qbert-v4 Seaquest-v4 SpaceInvaders-v4'.split()
        with pufferlib.utils.Suppress():
            envs = [gym.make(e) for e in env_ids]

    except:
        print_setup_error('Atari (ale)')
    else:
        return [
            pufferlib.binding.auto(
                env=e,
                env_name=env_id,
            )
            for env_id, e in zip(env_ids, envs)
        ]

def make_butterfly_bindings():
    try:
        from pettingzoo.butterfly import knights_archers_zombies_v8 as kaz
        from pettingzoo.butterfly import cooperative_pong_v5 as pong
        from pettingzoo.utils.conversions import aec_to_parallel_wrapper
    except:
        print_setup_error('Bufferfly (pettingzoo)')
    else:
        return [
            pufferlib.binding.auto(
                env_cls=aec_to_parallel_wrapper,
                env_args=[kaz.raw_env()],
                env_name='kaz',
            ),
            pufferlib.binding.auto(
                env_cls=aec_to_parallel_wrapper,
                env_args=[pong.raw_env()],
                env_name='cooperative-pong',
            ),
        ]

def make_classic_control_bindings():
    try:
        from gym.envs import classic_control
    except:
        print_setup_error('Classic Control (gym)')
    else:
        return pufferlib.binding.auto(
            env_cls=classic_control.CartPoleEnv,
            env_name='CartPole',
        )
 
def make_griddly_bindings():
    try:
        import griddly
        env_cls = lambda: gym.make('GDY-Spiders-v0')
        env_cls()
    except:
        print_setup_error('Spiders-v0 (griddly)')
    else:
        return pufferlib.binding.auto(
            env_cls=env_cls,
            obs_dtype=np.uint8,
        )

def make_magent_bindings():
    try:
        from pettingzoo.magent import battle_v3
        from pettingzoo.utils.conversions import aec_to_parallel_wrapper
    except:
        print_setup_error('MAgent (pettingzoo)')
    else:
        return pufferlib.binding.auto(
            env_cls=aec_to_parallel_wrapper,
            env_args=[battle_v3.env()],
            env_name='MAgent',
        )

def make_microrts_bindings():
    try:
        from gym_microrts.envs import GlobalAgentCombinedRewardEnv
    except:
        print_setup_error('Gym Microrts')
    else:
        return pufferlib.binding.auto(
            env_cls=GlobalAgentCombinedRewardEnv,
            env_name='Gym Microrts',
        )


def make_nethack_bindings():
    try:
        import nle
    except:
        print_setup_error('NetHack (nle)')
    else:
        return pufferlib.registry.NetHack()

def make_neuralmmo_bindings():
    try:
        import nmmo
    except:
        print_setup_error('Neural MMO (nmmo)')
    else:
        return pufferlib.binding.auto(
            env_cls=nmmo.Env,
            env_name='Neural MMO',
        )

def make_smac_bindings():
    try:
        from smac.env.pettingzoo.StarCraft2PZEnv import _parallel_env as smac_env
    except:
        print_setup_error('SMAC')
    else:
        return pufferlib.binding.auto(
            env_cls=smac_env,
            env_args=[1000],
            env_name='SMAC',
        )

def make_all_bindings():
    make_fns = [
        make_atari_bindings, # Well, almost all
        make_butterfly_bindings,
        make_classic_control_bindings,
        make_magent_bindings,
        make_nethack_bindings,
        make_neuralmmo_bindings,
        make_smac_bindings,
    ]

    bindings = {}
    for f in make_fns:
        bind = f()
        if bind is None:
            continue
        elif type(bind) != list:
            bind = [bind]

        for b in bind:
            assert b.env_name is not None
            assert b.env_name not in bindings, 'Duplicate env name'
            bindings[b.env_name] = b

    return pufferlib.utils.dotdict(bindings)