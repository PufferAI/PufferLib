import pufferlib
import pufferlib.emulation
import pufferlib.utils


def make_cartpole_binding():
    '''CartPole binding creation function
    
    This environment is a useful test because it works without
    any additional dependencies.'''
    try:
        from gym.envs import classic_control
    except:
        raise pufferlib.utils.SetupError('Classic Control (gym)')
    else:
        return pufferlib.emulation.Binding(
            env_cls=classic_control.CartPoleEnv,
            env_name='CartPole',
        )