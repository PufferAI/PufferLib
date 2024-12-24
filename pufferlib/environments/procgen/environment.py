from pdb import set_trace as T
import numpy as np

import gym
import gymnasium
import shimmy
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.postprocess

from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
)

def env_creator(name='bigfish'):
    return functools.partial(make, name)

def make(name, num_envs=1, num_levels=0, start_level=0,
        distribution_mode='easy', render_mode=None, buf=None):
    '''Atari creation function with default CleanRL preprocessing based on Stable Baselines3 wrappers'''
    assert int(num_envs) == float(num_envs), "num_envs must be an integer"
    num_envs = int(num_envs)

    procgen = pufferlib.environments.try_import('procgen')
    envs = procgen.ProcgenEnv(
        env_name=name,
        num_envs=num_envs,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        render_mode=render_mode,
    )
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    envs = ProcgenWrapper(envs)
    envs = shimmy.GymV21CompatibilityV0(env=envs, render_mode=render_mode)
    #envs = gymnasium.wrappers.GrayScaleObservation(envs)
    #envs = gymnasium.wrappers.FrameStack(envs, 4)#, framestack)
    #envs = MaxAndSkipEnv(envs, skip=2)
    envs = pufferlib.postprocess.EpisodeStats(envs)
    return pufferlib.emulation.GymnasiumPufferEnv(env=envs, buf=buf)

class ProcgenWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = self.env.observation_space['rgb']
        self.action_space = self.env.action_space

    @property
    def render_mode(self):
        return 'rgb_array'

    def reset(self, seed=None):
        obs = self.env.reset()[0]
        return obs

    def render(self, mode=None):
        return self.env.env.env.env.env.env.get_info()[0]['rgb']

    def close(self):
        return self.env.close()

    def step(self, actions):
        actions = np.asarray(actions).reshape(1)
        obs, rewards, dones, infos = self.env.step(actions)
        return obs[0], rewards[0], dones[0], infos[0]
