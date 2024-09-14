from pdb import set_trace as T
import numpy as np
import functools

import gym

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.utils
import pufferlib.postprocess
import pufferlib.wrappers

def env_creator(name='zelda'):
    if name == 'zelda':
        name = 'gvgai-zelda-lvl0-v0'
    return functools.partial(make, name)

def make(name, obs_type='grayscale', frameskip=4, full_action_space=False,
        repeat_action_probability=0.0, render_mode='rgb_array'):
    '''Atari creation function'''
    pufferlib.environments.try_import('gym_gvgai')
    env = gym.make(name)
    env = pufferlib.wrappers.GymToGymnasium(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    env = pufferlib.emulation.GymnasiumPufferEnv(env=env)
    return env

