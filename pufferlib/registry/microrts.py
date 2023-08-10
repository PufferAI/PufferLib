from pdb import set_trace as T

import pufferlib
import pufferlib.emulation
import pufferlib.exceptions
import pufferlib.models


Policy = pufferlib.models.Default



def make_env():
    '''Gym MicroRTS creation function
    
    This library appears broken. Step crashes in Java.
    '''
    try:
        from gym_microrts.envs import GlobalAgentCombinedRewardEnv
    except:
        raise pufferlib.exceptions.SetupError('microrts', 'GlobalAgentCombinedRewardEnv')
    else:
        return pufferlib.emulation.GymPufferEnv(
            env_creator=GlobalAgentCombinedRewardEnv,
        )