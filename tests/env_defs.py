from pdb import set_trace as T
import numpy as np

from dataclasses import dataclass

import gym
from gym.envs import classic_control, box2d
import griddly
import nle, nmmo
from pettingzoo.magent import battle_v3
from pettingzoo.butterfly import knights_archers_zombies_v8 as kaz
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
#from smac.env.pettingzoo.StarCraft2PZEnv import _parallel_env as smac_env

import pufferlib

#env_bindings = [pufferlib.bindings.auto(env_cls=nle.env.NLE)]

'''
@dataclass
class EnvSpec:
    cls: ...
    args: ... = []
    binding: ... = None

env_specs = {
    'nethack': EnvSpec(
        binding=pufferlib.bindings.NetHack()
    ),
    'nmmo': EnvSpec(
        cls=nmmo.Env
    ),
    'magent': EnvSpec(
        cls=aec_to_parallel_wrapper,
        args=[battle_v3.env()]
    ),
    'kaz': EnvSpec(
        cls=aec_to_parallel_wrapper,
        args=[kaz.raw_env()]
    ),
}

env_bindings = {}
for name, spec in env_specs.items():
    if spec.binding is None:
        env_bindings[name] = spec.binding
'''

'''
def make_griddly_env():
    import griddly, gym
    return gym.make('GDY-Spiders-v0')
    return lambda: gym.make(env_str)

bindings = [
    pufferlib.bindings.auto(
        env_cls=make_griddly_env,
        obs_dtype=np.uint8,
    )

]

'''
bindings = [
    pufferlib.bindings.auto(
        env_cls=classic_control.CartPoleEnv,
    ),
    pufferlib.bindings.auto(
        env=gym.make('Breakout-v4'),
    ),
    pufferlib.bindings.NetHack(),
    pufferlib.bindings.auto(
        env_cls=nmmo.Env,
    ),

    # RLlib error?
    pufferlib.bindings.auto(
        env_cls=aec_to_parallel_wrapper,
        env_args=[battle_v3.env()],
        env_name='magent',
    ),

    # Numpy versioning error?
    pufferlib.bindings.auto(
        env_cls=aec_to_parallel_wrapper,
        env_args=[kaz.raw_env()],
        env_name='kaz',
    ),
]

'''
    # SMAC will need a custom masked net
    pufferlib.bindings.auto(
        env_cls=smac_env,
        env_args=[1000],
        env_name='smac',
    )
'''

