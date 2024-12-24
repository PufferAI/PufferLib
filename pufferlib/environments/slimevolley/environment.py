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

def make(name, render_mode='rgb_array', buf=None):
    if name == 'slimevolley':
        name = 'SlimeVolley-v0'

    from slimevolleygym import SlimeVolleyEnv
    SlimeVolleyEnv.atari_mode = True
    env = SlimeVolleyEnv()
    env.policy.predict = lambda obs: np.random.randint(0, 2, 3)
    env = SlimeVolleyMultiDiscrete(env)
    env = SkipWrapper(env, repeat_count=4)
    env = shimmy.GymV21CompatibilityV0(env=env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

class SlimeVolleyMultiDiscrete(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        #self.action_space = gym.spaces.MultiDiscrete(
        #    [2 for _ in range(env.action_space.n)])

    def reset(self, seed=None):
        return self.env.reset().astype(np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs.astype(np.float32), reward, done, info

class SkipWrapper(gym.Wrapper):
    """
        Generic common frame skipping wrapper
        Will perform action for `x` additional steps
    """
    def __init__(self, env, repeat_count):
        super(SkipWrapper, self).__init__(env)
        self.repeat_count = repeat_count
        self.stepcount = 0

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < (self.repeat_count + 1) and not done:
            self.stepcount += 1
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1

        return obs, total_reward, done, info

    def reset(self):
        self.stepcount = 0
        return self.env.reset()
