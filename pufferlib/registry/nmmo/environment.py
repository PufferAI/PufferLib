from pdb import set_trace as T

import pufferlib
import pufferlib.emulation
import pufferlib.exceptions
from pufferlib.registry import try_import


def env_creator():
    nmmo = try_import('nmmo')
    return nmmo.Env

def make_env(*args, **kwargs):
    '''Neural MMO creation function'''
    env = env_creator()(*args, **kwargs)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)
