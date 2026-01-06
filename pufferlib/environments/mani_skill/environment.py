import functools
import numpy as np
from collections import defaultdict

import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import torch

import pufferlib

ALIASES = {
    'mani_pickcube': 'PickCube-v1',
    'mani_pushcube': 'PushCube-v1',
    'mani_stackcube': 'StackCube-v1',
    'mani_peginsertion': 'PegInsertionSide-v1',
}

def env_creator(name='PickCube-v1', **kwargs):
    return functools.partial(make, name)

def make(name, num_envs=1, render_mode='rgb_array', buf=None, seed=0, **kwargs):
    '''Create an environment by name'''

    if name in ALIASES:
        name = ALIASES[name]

    return ManiPufferEnv(name, num_envs=num_envs, render_mode=render_mode, buf=buf, seed=seed, **kwargs)

class ManiPufferEnv(pufferlib.PufferEnv):
    def __init__(self, name, num_envs=1, solver_position_iterations=15,
            sim_steps_per_control=5, control_freq=20, render_mode='rgb_array',
            log_interval=16, buf=None, seed=0):
        sim_freq = int(sim_steps_per_control * control_freq)
        sim_config = {
            'scene_config': {
                'solver_position_iterations': solver_position_iterations
            },
            'sim_freq': sim_freq,
            'control_freq': control_freq
        }
        self.env = gym.make(name, reward_mode='delta', num_envs=num_envs,
            render_mode=render_mode, sim_config=sim_config)
        self.env = ManiSkillVectorEnv(self.env, auto_reset=True, ignore_terminations=False, record_metrics=True)
        self.agents_per_batch = num_envs

        obs_space = self.env.observation_space
        self.single_observation_space = gym.spaces.Box(
            low=obs_space.low[0],
            high=obs_space.high[0],
            shape=obs_space.shape[1:],
            dtype=obs_space.dtype,
        )

        atn_space = self.env.action_space
        self.single_action_space = gym.spaces.Box(
            low=atn_space.low[0],
            high=atn_space.high[0],
            shape=atn_space.shape[1:],
            dtype=atn_space.dtype,
        )

        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.tick = 0

        self.env_id = np.arange(num_envs)

        self.logs = defaultdict(list)
        
        super().__init__(buf)

    def _flatten_info(self, info):
        if "final_info" in info:
            mask = info["_final_info"]
            for k, v in info["final_info"]["episode"].items():
                self.logs[k].append(v[mask].float().mean().item())

    def reset(self, seed=0):
        obs, info = self.env.reset()
        #self.observations = torch.nan_to_num(obs)
        self.observations = torch.clamp(torch.nan_to_num(obs), -5, 5)
        self.observations = obs / 20.0
        self._flatten_info(info)
        return obs, []

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        collapsed = torch.where(torch.isnan(obs).sum(1) > 0)[0]
        if len(collapsed) > 0:
            obs, _ = self.env.reset(options={'env_idx': collapsed})

        self.observations = torch.clamp(torch.nan_to_num(obs), -5, 5)
        #self.observations = obs / 20.0 #torch.nan_to_num(obs)
        self.rewards = reward
        self.terminated = terminated
        self.truncated = truncated
        self._flatten_info(info)

        self.infos = []
        self.tick += 1
        if self.tick % self.log_interval == 0:
            info = {}
            for k, v in self.logs.items():
                info[k] = np.mean(v)

            self.logs = defaultdict(list)
            self.infos.append(info)

        return obs, reward, terminated, truncated, self.infos

    def render(self):
        return self.env.render()[0].cpu().numpy()

    def close(self):
        self.env.close()
