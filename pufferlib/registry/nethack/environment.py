from pdb import set_trace as T

import pufferlib
import pufferlib.emulation
import pufferlib.utils


def env_creator():
    try:
        import nle
    except:
        raise pufferlib.utils.SetupError('NetHack (nle)')
    else:
        return nle.env.NLE
 
def make_env():
    '''NetHack binding creation function'''
    env = env_creator()()
    return pufferlib.emulation.GymPufferEnv(env=env)
