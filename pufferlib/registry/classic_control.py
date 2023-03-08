import pufferlib
import pufferlib.emulation
import pufferlib.utils


def make_cartpole_binding():
    try:
        from gym.envs import classic_control
    except:
        raise pufferlib.utils.SetupError('Classic Control (gym)')
    else:
        return pufferlib.emulation.Binding(
            env_cls=classic_control.CartPoleEnv,
            env_name='CartPole',
        )