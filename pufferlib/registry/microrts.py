import pufferlib
import pufferlib.emulation
import pufferlib.utils


def make_env():
    '''Gym MicroRTS binding creation function
    
    Currently only provides a binding for the GlobalAgentCombinedRewardEnv
    setting of the environment.'''
    try:
        from gym_microrts.envs import GlobalAgentCombinedRewardEnv
    except:
        raise pufferlib.utils.SetupError('Gym MicroRTS')
    else:
        return pufferlib.emulation.PettingZooPufferEnv(
            env_creator=GlobalAgentCombinedRewardEnv,
            #env_name='Gym MicroRTS',
        )