from pdb import set_trace as T

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.wrappers


def env_creator():
    nmmo = pufferlib.environments.try_import('nmmo')
    return nmmo.Env

def make_env(*args, **kwargs):
    '''Neural MMO creation function'''
    env = env_creator()(*args, **kwargs)
    env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)
