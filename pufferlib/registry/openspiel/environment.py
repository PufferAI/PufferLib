from pdb import set_trace as T

import gym
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
import shimmy

import pufferlib
import pufferlib.emulation
import pufferlib.registry


def env_creator():
    '''OpenSpiel creation function'''
    pyspiel = pufferlib.registry.try_import('pyspiel', 'openspiel')
    return pyspiel.load_game

def make_env(name='2048'):
    '''OpenSpiel creation function'''
    env = env_creator()(name)
    env = shimmy.OpenspielCompatibilityV0(env, None)
    # TODO: needs custom conversion to parallel
    env = aec_to_parallel_wrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)
