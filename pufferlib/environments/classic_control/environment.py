import gymnasium
from gymnasium.envs import classic_control
import functools
import numpy as np

import pufferlib
import pufferlib.emulation
import pufferlib.postprocess

ALIASES = {
    'cartpole': 'CartPole-v0',
    'mountaincar': 'MountainCar-v0',
}

def env_creator(name='cartpole'):
    return functools.partial(make, name)

def make(name, render_mode='rgb_array', buf=None):
    '''Create an environment by name'''

    if name in ALIASES:
        name = ALIASES[name]

    env = gymnasium.make(name, render_mode=render_mode)
    if name == 'MountainCar-v0':
        env = MountainCarWrapper(env)

    #env = gymnasium.wrappers.NormalizeObservation(env)
    env = gymnasium.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -1, 1))
    #env = gymnasium.wrappers.NormalizeReward(env, gamma=gamma)
    env = gymnasium.wrappers.TransformReward(env, lambda reward: np.clip(reward, -1, 1))
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

class MountainCarWrapper(gymnasium.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = abs(obs[0]+0.5)
        return obs, reward, terminated, truncated, info
