import pufferlib
import pufferlib.emulation
import pufferlib.utils


def make_binding():
    '''Starcraft Multiagent Challenge binding creation function

    Support for SMAC is WIP because environments do not function without
    an action-masked baseline policy.'''
    try:
        from smac.env.pettingzoo.StarCraft2PZEnv import _parallel_env as smac_env
    except:
        raise pufferlib.utils.SetupError('SMAC')
    else:
        return pufferlib.emulation.PettingZooPufferEnv(
            env_creator=smac_env,
            env_args=[1000],
            #env_name='SMAC',
        )