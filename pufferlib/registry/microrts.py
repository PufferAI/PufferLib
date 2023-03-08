import pufferlib
import pufferlib.emulation
import pufferlib.utils


def make_binding():
    try:
        from gym_microrts.envs import GlobalAgentCombinedRewardEnv
    except:
        raise pufferlib.utils.SetupError('Gym MicroRTS')
    else:
        return pufferlib.emulation.Binding(
            env_cls=GlobalAgentCombinedRewardEnv,
            env_name='Gym MicroRTS',
        )