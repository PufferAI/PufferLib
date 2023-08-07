import gym

import pufferlib
import pufferlib.emulation
import pufferlib.utils


def make_env(name='DeepmindLabSeekavoidArena01-v0'):
    '''Deepmind Lab binding creation function
    dm-lab requires extensive setup. Use PufferTank.'''
    try:
        import gym_deepmindlab
    except:
        raise pufferlib.utils.SetupError('Deepmind Lab (dm-lab)')
    else:
        return pufferlib.emulation.GymPufferEnv(
            env_creator=gym.make,
            env_args=[name],
        )