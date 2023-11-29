import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.wrappers


def env_creator():
    pufferlib.environments.try_import('smac')
    from smac.env.pettingzoo.StarCraft2PZEnv import _parallel_env as smac_env
    return smac_env
 
def make_env():
    '''Starcraft Multiagent Challenge creation function

    Support for SMAC is WIP because environments do not function without
    an action-masked baseline policy.'''
    env = env_creator()(1000)
    env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
    env = pufferlib.emulation.PettingZooPufferEnv(env)
    return env
