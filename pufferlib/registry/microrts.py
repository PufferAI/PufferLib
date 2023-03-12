import pufferlib
import pufferlib.emulation
import pufferlib.utils


def make_binding():
    '''Gym MicroRTS binding creation function
    
    Currently only provides a binding for the GlobalAgentCombinedRewardEnv
    setting of the environment.'''
    try:
        from gym_microrts.envs import GlobalAgentCombinedRewardEnv
    except:
        raise pufferlib.utils.SetupError('Gym MicroRTS')
    else:
        return pufferlib.emulation.Binding(
            env_cls=GlobalAgentCombinedRewardEnv,
            env_name='Gym MicroRTS',
        )