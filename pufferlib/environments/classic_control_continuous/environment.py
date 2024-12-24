import gymnasium
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.postprocess


def env_creator(name='MountainCarContinuous-v0'):
    return functools.partial(make, name)

def make(name, render_mode='rgb_array', buf=None):
    '''Create an environment by name'''
    env = gymnasium.make(name, render_mode=render_mode)
    if name == 'MountainCarContinuous-v0':
        env = MountainCarWrapper(env)

    env = pufferlib.postprocess.ClipAction(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

class MountainCarWrapper(gymnasium.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = abs(obs[0]+0.5)
        return obs, reward, terminated, truncated, info
