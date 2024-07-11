from pdb import set_trace as T

import gymnasium
import functools

import pufferlib.emulation
import pufferlib.environments
import pufferlib.postprocess


def env_creator(name='CarRacing-v2'):
    return functools.partial(make, name=name)

def make(name, domain_randomize=True, continuous=False, render_mode='rgb_array'):
    #try:
    env = gymnasium.make(name, render_mode=render_mode,
        domain_randomize=domain_randomize, continuous=continuous)
    #except:
    #    raise ValueError(
    #        f'Env {name} not found or not installed. Try pip install pufferlib[box2d]')
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)
