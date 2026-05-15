
import gymnasium

from links_awaken import LinksAwakenV1 as env_creator

import pufferlib.emulation


def make_env(headless: bool = True, state_path=None, buf=None):
    '''Links Awakening'''
    env = env_creator(headless=headless, state_path=state_path)
    env = gymnasium.wrappers.ResizeObservation(env, shape=(72, 80))
    return pufferlib.emulation.GymnasiumPufferEnv(env=env,
        postprocessor_cls=pufferlib.emulation.BasicPostprocessor, buf=buf)
