from gymnasium.envs import classic_control
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.postprocess


def env_creator(name='cartpole'):
    return functools.partial(make, name)

def make(name, render_mode='rgb_array'):
    '''Create an environment by name'''
    if name == 'cartpole':
        env_cls = classic_control.CartPoleEnv
    else:
        raise ValueError(f'Unknown environment: {name}')

    env = env_cls(render_mode=render_mode)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)
