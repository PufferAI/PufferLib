import gym

import pufferlib
import pufferlib.emulation
import pufferlib.exceptions
import pufferlib.models


Policy = pufferlib.models.Default

def make_env(name='MineRLNavigateDense-v0'):
    '''Minecraft environment creation function

    Currently broken: requires Gym 0.19, which
    is a broken package 
    '''
    try:
        import minerl
    except:
        raise pufferlib.exceptions.SetupError('minerl', name)
    else:
        return pufferlib.emulation.GymPufferEnv(
            env_creator=gym.make,
            env_args=[name],
        )