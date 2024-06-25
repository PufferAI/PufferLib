from pdb import set_trace as T
import numpy as np
import functools

import gym
import shimmy

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.utils
import pufferlib.postprocess


def env_creator(name='SlimeVolley-v0'):
    return functools.partial(make, name)

def make(name, render_mode='rgb_array'):
    import slimevolley
    env = gym.make(name)
    env = shimmy.GymV21CompatibilityV0(env=env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)
