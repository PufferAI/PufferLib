from pdb import set_trace as T
import numpy as np
import pettingzoo
import gymnasium as gym
import functools
import yaml
import os

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.wrappers
import pufferlib.postprocess


def env_creator(name='nmmo'):
    return functools.partial(make, name)

def make(name, *args, **kwargs):
    '''Neural MMO creation function'''
    path = os.path.dirname(os.path.realpath(__file__))
    f_path = os.path.join(path, 'env_config.yaml')
    with open(f_path, "r") as stream:
        env_config = yaml.safe_load(stream)

    from nocturne.envs.base_env import BaseEnv
    env = BaseEnv(config=env_config)
    env.files = env.files[:1]
    env = NocturneWrapper(env)
    env = pufferlib.postprocess.MultiagentEpisodeStats(env)
    env = pufferlib.postprocess.MeanOverAgents(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)

class NocturneWrapper(pettingzoo.ParallelEnv):
    def __init__(self, env):
        self.env = env
        obs = env.reset()

        self.possible_agents = list(obs.keys())
        self.agents = list(obs.keys())
        self.empty_infos = {a: {} for a in self.agents}
        self.truncations = {a: False for a in self.agents}

    def observation_space(self, agent):
        return self.env.observation_space

    def action_space(self, agent):
        return self.env.action_space

    def reset(self, seed=None):
        obs = self.env.reset()
        obs = {k: v.astype(np.float32) for k, v in obs.items()}
        return obs, self.empty_infos

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        obs = {k: v.astype(np.float32) for k, v in obs.items()}
        return obs, rewards, dones, self.truncations, infos

    def close(self):
        self.env.close()
