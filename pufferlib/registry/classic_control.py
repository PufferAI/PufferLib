import pufferlib
import pufferlib.emulation
import pufferlib.utils


def make_cartpole_env():
    '''CartPole binding creation function
    
    This environment is a useful test because it works without
    any additional dependencies.'''
    try:
        from gym.envs import classic_control
    except:
        raise pufferlib.utils.SetupError('Classic Control (gym)')
    else:
        return pufferlib.emulation.GymPufferEnv(
            env=classic_control.CartPoleEnv()
        )