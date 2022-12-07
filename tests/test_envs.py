from pdb import set_trace as T

import nle, nmmo
from pettingzoo.magent import battle_v3
from pettingzoo.butterfly import knights_archers_zombies_v8 as kaz
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
#from smac.env.pettingzoo.StarCraft2PZEnv import _parallel_env as smac_env

import pufferlib


env_bindings = [
    pufferlib.bindings.NetHack(),
    pufferlib.bindings.auto(
        env_cls=nmmo.Env
    ),
]
'''
    # RLlib error
    pufferlib.bindings.auto(
        env_cls=aec_to_parallel_wrapper,
        env_args=[battle_v3.env()],
        env_name='magent',
    ),

    # Numpy versioning error
    pufferlib.bindings.auto(
        env_cls=aec_to_parallel_wrapper,
        env_args=[kaz.raw_env()],
        env_name='kaz',
    ),
    # SMAC will need a custom masked net
    pufferlib.bindings.auto(
        env_cls=smac_env,
        env_args=[1000],
        env_name='smac',
    )
'''

#env_bindings = [pufferlib.bindings.auto(env_cls=nle.env.NLE)]
for binding in env_bindings:
    tuner = pufferlib.rllib.make_rllib_tuner(binding)
    result = tuner.fit()[0]
    print('Saved ', result.checkpoint)
