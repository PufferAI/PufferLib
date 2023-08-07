import gym

import pufferlib
import pufferlib.emulation
import pufferlib.utils

def make_env(name='MineRLNavigateDense-v0'):
    '''Minecraft environment creation function

    No support for bindings yet because MineRL requires
    a very old version of Gym
    
    TODO: Add support for Gym 0.19
    '''
    try:
        import minerl
    except:
        raise pufferlib.utils.SetupError('Minecraft (MineRL)')
    else:
        return gym.make(name)