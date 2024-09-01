from pdb import set_trace as T

import gym
import shimmy
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.postprocess

ALIASES = {
    'spiders': 'GDY-Spiders-v0',
}    

def env_creator(name='spiders'):
    return functools.partial(make, name)

# TODO: fix griddly
def make(name):
    '''Griddly creation function

    Note that Griddly environments do not have observation spaces until
    they are created and reset'''
    if name in ALIASES:
        name = ALIASES[name]

    pufferlib.environments.try_import('griddly')
    with pufferlib.utils.Suppress():
        env = gym.make(name)
        env.reset() # Populate observation space

    env = shimmy.GymV21CompatibilityV0(env=env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env)
