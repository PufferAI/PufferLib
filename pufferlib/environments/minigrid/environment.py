from pdb import set_trace as T

import gymnasium
import functools

import pufferlib.emulation
import pufferlib.environments
import pufferlib.postprocess


def env_creator(name='MiniGrid-LavaGapS7-v0'):
    return functools.partial(make, name=name)

def make(name, render_mode='rgb_array'):
    minigrid = pufferlib.environments.try_import('minigrid')
    env = gymnasium.make(name, render_mode=render_mode)
    env = MiniGridWrapper(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

class MiniGridWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = gymnasium.spaces.Dict({
            k: v for k, v in self.env.observation_space.items() if
            k != 'mission'
        })
        self.action_space = self.env.action_space
        self.close = self.env.close
        self.render = self.env.render
        self.close = self.env.close
        self.render_mode = 'rgb_array'

    def reset(self, seed=None):
        self.tick = 0
        obs, info = self.env.reset(seed=seed)
        del obs['mission']
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        del obs['mission']

        self.tick += 1
        if self.tick == 100:
            done = True

        return obs, reward, done, truncated, info
