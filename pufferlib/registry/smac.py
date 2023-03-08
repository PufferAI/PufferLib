import pufferlib
import pufferlib.emulation
import pufferlib.utils


def make_binding():
    try:
        from smac.env.pettingzoo.StarCraft2PZEnv import _parallel_env as smac_env
    except:
        raise pufferlib.utils.SetupError('SMAC')
    else:
        return pufferlib.emulation.Binding(
            env_cls=smac_env,
            default_args=[1000],
            env_name='SMAC',
        )